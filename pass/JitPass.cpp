//=============================================================================
// FILE:
//    JitPass.cpp
//
// DESCRIPTION:
//    Find functions annotated with "jit" plus input arguments that are
//    amenable to runtime constant propagation. Stores the IR for those
//    functions, replaces them with a stub function that calls the jit runtime
//    library to compile the IR and call the function pointer of the jit'ed
//    version.
//
// USAGE:
//    1. Legacy PM
//      opt -enable-new-pm=0 -load libJitPass.dylib -legacy-jit-pass
//      -disable-output `\`
//        <input-llvm-file>
//    2. New PM
//      opt -load-pass-plugin=libJitPass.dylib -passes="jit-pass" `\`
//        -disable-output <input-llvm-file>
//
//
// License: MIT
//=============================================================================
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/ExecutionEngine/Orc/IndirectionUtils.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Object/ELF.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/IPO/StripDeadPrototypes.h"
#include "llvm/Transforms/IPO/StripSymbols.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include <llvm/ADT/StringRef.h>
#include <llvm/CodeGen/Register.h>
#include <llvm/IR/CallingConv.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/GlobalValue.h>

#include <iostream>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/MemoryBuffer.h>
#include <string>

// #define ENABLE_RECURSIVE_JIT
#define DEBUG_TYPE "jitpass"
#ifdef ENABLE_DEBUG
#define DEBUG(x) x
#else
#define DEBUG(x)
#endif

using namespace llvm;

//-----------------------------------------------------------------------------
// JitPass implementation
//-----------------------------------------------------------------------------
// No need to expose the internals of the pass to the outside world - keep
// everything in an anonymous namespace.
namespace {

struct JitFunctionInfo {
  Function *Fn;
  SmallVector<int, 8> ConstantArgs;
  std::string ModuleIR;
};

SmallVector<JitFunctionInfo, 8> JitFunctionInfoList;

static bool isDeviceCompilation(Module &M) {
  Triple TargetTriple(M.getTargetTriple());
  dbgs() << "TargetTriple " << M.getTargetTriple() << "\n";
  if (TargetTriple.isNVPTX() || TargetTriple.isAMDGCN())
    return true;

  return false;
}

static bool isDeviceKernel(Module &M, Function &Fn, CallBase **LaunchKernelCall,
                           Value **Kernel) {
  // TODO: generalize to CUDA.
  Function *LaunchKernelFn = M.getFunction("hipLaunchKernel");
  *LaunchKernelCall = nullptr;
  *Kernel = nullptr;
  if (!LaunchKernelFn)
    return false;

  for (User *Usr : LaunchKernelFn->users())
    if (CallBase *CB = dyn_cast<CallBase>(Usr))
      if (CB->getFunction() == &Fn) {
        dbgs() << "Found kernel stub " << Fn.getName() << "\n";
        *LaunchKernelCall = CB;
        *Kernel = CB->getOperand(0);
        return true;
      }

  return false;
}

static bool HasDeviceKernels(Module &M) {
  // TODO: generalize to CUDA.
  Function *LaunchKernelFn = M.getFunction("hipLaunchKernel");
  if (!LaunchKernelFn)
    return false;

  return true;
}

void parseAnnotations(Module &M) {
  auto GlobalAnnotations = M.getNamedGlobal("llvm.global.annotations");
  if (GlobalAnnotations) {
    dbgs() << "isDevice " << isDeviceCompilation(M) << "\n";
    auto Array = cast<ConstantArray>(GlobalAnnotations->getOperand(0));
    dbgs() << "Array " << *Array << "\n";
    for (int i = 0; i < Array->getNumOperands(); i++) {
      auto Entry = cast<ConstantStruct>(Array->getOperand(i));
      dbgs() << "Entry " << *Entry << "\n";

      auto Fn = dyn_cast<Function>(Entry->getOperand(0)->stripPointerCasts());

      assert(Fn && "Expected function in entry operands");

      for (auto &JFI : JitFunctionInfoList)
        if (JFI.Fn == Fn)
          report_fatal_error("Duplicate jit annotation for Fn " + Fn->getName(),
                             false);

      dbgs() << "JIT Function " << Fn->getName() << "\n";

      auto Annotation =
          cast<ConstantDataArray>(Entry->getOperand(1)->getOperand(0));

      dbgs() << "Annotation " << Annotation->getAsCString() << "\n";

      // TODO: needs CString for comparison to work, why?
      if (Annotation->getAsCString().compare("jit"))
        continue;

      JitFunctionInfo JFI;
      JFI.Fn = Fn;

      if (Entry->getOperand(4)->isNullValue())
        JFI.ConstantArgs = {};
      else {
        dbgs() << "AnnotArgs " << *Entry->getOperand(4)->getOperand(0) << "\n";
        dbgs() << "Type AnnotArgs "
               << *Entry->getOperand(4)->getOperand(0)->getType() << "\n";
        auto AnnotArgs =
            cast<ConstantStruct>(Entry->getOperand(4)->getOperand(0));

        for (int I = 0; I < AnnotArgs->getNumOperands(); ++I) {
          auto *Index = cast<ConstantInt>(AnnotArgs->getOperand(I));
          // TODO: think about types, check within function arguments bounds, -1
          // to convert to 0-start index.
          JFI.ConstantArgs.push_back(Index->getValue().getZExtValue() - 1);
        }
      }

      JitFunctionInfoList.push_back(JFI);
    }
  }
}

static void
getReachableFunctions(Module &M, Function &F,
                      SmallPtrSetImpl<Function *> &ReachableFunctions) {
  SmallVector<Function *, 8> ToVisit;
  ToVisit.push_back(&F);
  CallGraphWrapperPass CG;
  CG.runOnModule(M);
  while (!ToVisit.empty()) {
    Function *VisitF = ToVisit.pop_back_val();
    CallGraphNode *CGNode = CG[VisitF];

    for (const auto &Callee : *CGNode) {
      Function *CalleeF = Callee.second->getFunction();

      if (!CalleeF) {
        dbgs() << "Skip external node\n";
        continue;
      }

      if (CalleeF->isDeclaration()) {
        dbgs() << "Skip declaration of " << CalleeF->getName() << "\n";
        continue;
      }

      if (ReachableFunctions.contains(CalleeF)) {
        dbgs() << "Skip already visited " << CalleeF->getName() << "\n";
        continue;
      }

      dbgs() << "Found reachable " << CalleeF->getName() << " ... to visit\n";
      ReachableFunctions.insert(CalleeF);
      ToVisit.push_back(CalleeF);
    }
  }
}

void runCleanupPassPipeline(Module &M) {
  PassBuilder PB;
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  ModulePassManager Passes;
  // Using pass for extraction (to remove?)
  // Passes.add(createGVExtractionPass(GlobalsToExtract));
  Passes.addPass(AlwaysInlinerPass());
  Passes.addPass(GlobalDCEPass());
  Passes.addPass(StripDeadDebugInfoPass());
  Passes.addPass(StripDeadPrototypesPass());
  // JitMod
  Passes.run(M, MAM);
}

static void createJITModule(Module &M, JitFunctionInfo &JFI) {
  SmallPtrSet<Function *, 16> ReachableFunctions;
  getReachableFunctions(M, *JFI.Fn, ReachableFunctions);
  ReachableFunctions.insert(JFI.Fn);

  ValueToValueMapTy VMap;
  auto JitMod =
      CloneModule(M, VMap, [&ReachableFunctions, &JFI](const GlobalValue *GV) {
        if (const GlobalVariable *G = dyn_cast<GlobalVariable>(GV)) {
          if (!G->isConstant())
            return false;
          // For constant global variables, keep their definitions only
          // if they are reachable by any of the functions in the
          // JIT module.
          // TODO: Is isConstant() enough? Maybe we want isManifestConstant()
          // that makes sure that the constant is free of unknown values.
          for (const User *Usr : GV->users()) {
            const Instruction *UsrI = dyn_cast<Instruction>(Usr);
            if (!UsrI)
              continue;
            const Function *ParentF = UsrI->getParent()->getParent();
            if (ReachableFunctions.contains(ParentF))
              return true;
          }

          return false;
        }

        // TODO: do not clone aliases' definitions, it this sound?
        if (const GlobalAlias *GA = dyn_cast<GlobalAlias>(GV))
          return false;

        if (const Function *OrigF = dyn_cast<Function>(GV)) {
          if (OrigF == JFI.Fn) {
            dbgs() << "OrigF " << OrigF->getName()
                   << " == " << JFI.Fn->getName() << ", definitely keep\n";
            return true;
          }
          // Do not keep definitions of unreachable functions.
          if (!ReachableFunctions.contains(OrigF)) {
            // dbgs() << "Drop unreachable " << F->getName() << "\n";
            return false;
          }

#ifdef ENABLE_RECURSIVE_JIT
          // Enable recursive jit'ing.
          for (auto &JFIInner : JitFunctionInfoList)
            if (JFIInner.Fn == OrigF) {
              dbgs() << "Do not keep definitions of another jit function "
                     << OrigF->getName() << "\n";
              return false;
            }
#endif

          // dbgs() << "Keep reachable " << F->getName() << "\n";

          return true;
        }

        // For the rest global values do not keep their definitions.
        return false;
      });

  Function *JitF = cast<Function>(VMap[JFI.Fn]);
  JitF->setLinkage(GlobalValue::ExternalLinkage);
  // Run a global DCE pass on the JIT module IR to remove
  // unnecessary symbols and reduce the IR to JIT at runtime.
  DEBUG(dbgs() << "=== Pre DCE JIT IR\n"
               << *JitMod << "=== End of Pre DCE JIT IR\n");
  // Using pass for extraction (to remove?)
  // std::vector<GlobalValue *> GlobalsToExtract;
  // for (auto *F : ReachableFunctions) {
  //  GlobalValue *GV = dyn_cast<GlobalValue>(JitF);
  //  assert(GV && "Expected non-null GV");
  //  GlobalsToExtract.push_back(GV);
  //  dbgs() << "Extract global " << *GV << "\n";
  //}

  // Internalize functions, besides JIT function, in the module
  // to inline.
  for (Function &JitModF : *JitMod) {
    if (JitModF.isDeclaration())
      continue;

    if (&JitModF == JitF)
      continue;

    // Internalize other functions in the module.
    JitModF.setLinkage(GlobalValue::InternalLinkage);
    // F.setLinkage(GlobalValue::PrivateLinkage);
    JitModF.removeFnAttr(Attribute::NoInline);
    // F.addFnAttr(Attribute::InlineHint);
    JitModF.addFnAttr(Attribute::AlwaysInline);
  }

  DEBUG(dbgs() << "=== Pre Passes JIT IR\n"
               << *JitMod << "=== End of Pre Passes JIT R\n");

  runCleanupPassPipeline(*JitMod);

  DEBUG(dbgs() << "=== Post Passes JIT IR\n"
               << *JitMod << "=== End of Post Passes JIT R\n");

  // Update linkage and visibility in the original module only for
  // globals included in the JIT module required for external
  // linking.
  for (auto &GVar : M.globals()) {
    if (VMap[&GVar]) {
      dbgs() << "=== GVar\n";
      dbgs() << GVar << "\n";
      dbgs() << "Linkage " << GVar.getLinkage() << "\n";
      dbgs() << "Visibility " << GVar.getVisibility() << "\n";
      dbgs() << "=== End GV\n";

      if (GVar.isConstant())
        continue;

      if (GVar.getName() == "llvm.global_ctors") {
        dbgs() << "Skip llvm.global_ctors";
        continue;
      }

      if (GVar.hasAvailableExternallyLinkage()) {
        dbgs() << "Skip available externally";
        continue;
      }

      GVar.setLinkage(GlobalValue::ExternalLinkage);
      GVar.setVisibility(GlobalValue::VisibilityTypes::DefaultVisibility);
    }
  }

#ifdef ENABLE_RECURSIVE_JIT
  // Set linkage to external for any reachable jit'ed function to enable
  // recursive jit'ing.
  for (auto &JFIInner : JitFunctionInfoList) {
    if (!ReachableFunctions.contains(JFIInner.Fn))
      continue;
    if (VMap[JFIInner.Fn]) {
      Function *JitF = cast<Function>(VMap[JFIInner.Fn]);
      JFIInner.Fn->setLinkage(GlobalValue::ExternalLinkage);
      JFIInner.Fn->setVisibility(
          GlobalValue::VisibilityTypes::DefaultVisibility);
    }
  }
#endif
  // TODO: Do we want to keep debug info to use for GDB/LLDB
  // interfaces for debugging jitted code?
  StripDebugInfo(*JitMod);

  // Add metadata for the JIT function to store the argument positions for
  // runtime constants.
  LLVMContext &Ctx = JitMod->getContext();
  SmallVector<Metadata *> ConstArgNos;
  for (size_t I = 0; I < JFI.ConstantArgs.size(); ++I) {
    int ArgNo = JFI.ConstantArgs[I];
    Metadata *Meta =
        ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(Ctx), ArgNo));
    ConstArgNos.push_back(Meta);
  }
  MDNode *Node = MDNode::get(JitMod->getContext(), ConstArgNos);
  JitF->setMetadata("jit_arg_nos", Node);

  if (verifyModule(*JitMod, &errs()))
    report_fatal_error("Broken JIT module found, compilation aborted!", false);
  else
    dbgs() << "JitMod verified!\n";

  raw_string_ostream OS(JFI.ModuleIR);
  WriteBitcodeToFile(*JitMod, OS);
  OS.flush();

  DEBUG(dbgs() << "=== Post DCE JIT IR\n"
               << *JitMod << "=== End of Post DCE JIT IR\n");
}

static void createJITModuleSectionsDevice(Module &M, JitFunctionInfo &JFI) {
  ValueToValueMapTy VMap;
  // TODO: We clone everything, use ReachableFunctions to prune. What happens if
  // there are cross-kernel globals?
  // Need to remove all other __global__ functions in the module,
  // hipModuleLoadData expects a single kernel (__global__) in the image.
  auto JitMod = CloneModule(M, VMap, [&JFI](const GlobalValue *GV) {
    // Do not clone JIT bitcodes of other kernels.
    if (GV->getSection().starts_with(".jit."))
      return false;

    if (const Function *F = dyn_cast<Function>(GV)) {
      if (F == JFI.Fn)
        return true;
      // Do not clone other host-callable kernels.
      // TODO: Is this necessary when using hiprtc? In any case it reduces the
      // JITted code which should reduce compilation time.
      if (F->getCallingConv() == CallingConv::AMDGPU_KERNEL)
        return false;
    }

    return true;
  });

  // Remove llvm.global.annotations and .jit section globals from the module and
  // used lists.
  SmallPtrSet<GlobalVariable *, 8> JitGlobalsToRemove;
  auto JitGlobalAnnotations = JitMod->getNamedGlobal("llvm.global.annotations");
  assert(JitGlobalAnnotations &&
         "Expected llvm.global.annotations in jit module");
  JitGlobalsToRemove.insert(JitGlobalAnnotations);

  for (auto &GV : JitMod->globals()) {
    if (GV.getSection().starts_with(".jit."))
      JitGlobalsToRemove.insert(&GV);
  }

  removeFromUsedLists(*JitMod, [&JitGlobalsToRemove](Constant *C) {
    if (auto *GV = dyn_cast<GlobalVariable>(C)) {
      if (JitGlobalsToRemove.contains(GV))
        return true;
    }

    return false;
  });

  for (auto *GV : JitGlobalsToRemove)
    JitMod->eraseGlobalVariable(GV);

  Function *JitF = cast<Function>(VMap[JFI.Fn]);
  // Run a global DCE pass on the JIT module IR to remove
  // unnecessary symbols and reduce the IR to JIT at runtime.
  DEBUG(dbgs() << "=== Pre DCE JIT IR\n"
               << *JitMod << "=== End of Pre DCE JIT IR\n");

  // TODO: is this necessary? Other tools could be performing it already.
  // Internalize functions, besides JIT function, in the module
  // to inline.
  for (Function &JitModF : *JitMod) {
    if (JitModF.isDeclaration())
      continue;

    if (&JitModF == JitF)
      continue;

    // Internalize other functions in the module.
    JitModF.setLinkage(GlobalValue::InternalLinkage);
    // F.setLinkage(GlobalValue::PrivateLinkage);
    JitModF.removeFnAttr(Attribute::NoInline);
    // F.addFnAttr(Attribute::InlineHint);
    JitModF.addFnAttr(Attribute::AlwaysInline);
  }

  DEBUG(dbgs() << "=== Pre Passes JIT IR\n"
               << *JitMod << "=== End of Pre Passes JIT R\n");

  runCleanupPassPipeline(*JitMod);

  DEBUG(dbgs() << "=== Post Passes JIT IR\n"
               << *JitMod << "=== End of Post Passes JIT R\n");

  // TODO: Do we want to keep debug info to use for GDB/LLDB
  // interfaces for debugging jitted code?
  StripDebugInfo(*JitMod);

  // Add metadata for the JIT function to store the argument positions for
  // runtime constants.
  LLVMContext &Ctx = JitMod->getContext();
  SmallVector<Metadata *> ConstArgNos;
  for (size_t I = 0; I < JFI.ConstantArgs.size(); ++I) {
    int ArgNo = JFI.ConstantArgs[I];
    Metadata *Meta =
        ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(Ctx), ArgNo));
    ConstArgNos.push_back(Meta);
  }
  MDNode *Node = MDNode::get(JitMod->getContext(), ConstArgNos);
  JitF->setMetadata("jit_arg_nos", Node);

  if (verifyModule(*JitMod, &errs()))
    report_fatal_error("Broken JIT module found, compilation aborted!", false);
  else
    dbgs() << "JitMod verified!\n";

  raw_string_ostream OS(JFI.ModuleIR);
  WriteBitcodeToFile(*JitMod, OS);
  OS.flush();

  DEBUG(dbgs() << "=== Post DCE JIT IR\n"
               << *JitMod << "=== End of Post DCE JIT IR\n");

  dbgs() << "Create JIT sections\n";
  Constant *JitModule = ConstantDataArray::get(
      M.getContext(), ArrayRef<uint8_t>((const uint8_t *)JFI.ModuleIR.data(),
                                        JFI.ModuleIR.size()));
  auto *GV = new GlobalVariable(M, JitModule->getType(), /* isConstant */ true,
                                GlobalValue::PrivateLinkage, JitModule,
                                ".jit.bc." + JFI.Fn->getName());
  appendToUsed(M, {GV});
  GV->setSection(Twine(".jit." + JFI.Fn->getName()).str());
  std::error_code EC;
  raw_fd_ostream Output(Twine("jit-" + JFI.Fn->getName() + ".bc").str(), EC);
  Output << JFI.ModuleIR;
  Output.close();

  return;
}

static void emitJITKernelStub(Module &M, JitFunctionInfo &JFI,
                              StringRef KernelName,
                              CallBase *LaunchKernelCall) {
  // Create jit entry runtime function.
  Type *VoidPtrTy = Type::getInt8PtrTy(M.getContext());
  Type *Int64Ty = Type::getInt64Ty(M.getContext());
  Type *Int32Ty = Type::getInt32Ty(M.getContext());
  StructType *RuntimeConstantTy = StructType::create({Int64Ty}, "struct.args");

  GlobalVariable *HipFatbinWrapper =
      M.getGlobalVariable("__hip_fatbin_wrapper", true);
  assert(HipFatbinWrapper && "Expected hip fatbinary wrapper global");

  FunctionType *JitEntryFnTy = FunctionType::get(
      Int32Ty,
      {VoidPtrTy, HipFatbinWrapper->getType(),
       RuntimeConstantTy->getPointerTo(), Int32Ty, Int64Ty, Int32Ty, Int64Ty,
       Int32Ty, VoidPtrTy, Int64Ty, VoidPtrTy},
      /* isVarArg=*/false);
  FunctionCallee JitEntryFn =
      M.getOrInsertFunction("__jit_launch_kernel", JitEntryFnTy);

  // Insert before the launch kernel call instruction.
  IRBuilder<> Builder(LaunchKernelCall);

  // Create globals for the function name and string IR passed to the jit
  // entry.
  auto *FnNameGlobal = Builder.CreateGlobalString(KernelName);

  // Create the runtime constant array type for the runtime constants passed
  // to the jit entry function.
  ArrayType *RuntimeConstantArrayTy =
      ArrayType::get(RuntimeConstantTy, JFI.ConstantArgs.size());

  // Create the runtime constants data structure passed to the jit entry.
  Value *RuntimeConstantsAlloca = nullptr;
  if (JFI.ConstantArgs.size() > 0) {
    auto IP = Builder.saveIP();
    Builder.SetInsertPoint(&JFI.Fn->getEntryBlock(),
                           JFI.Fn->getEntryBlock().getFirstInsertionPt());
    RuntimeConstantsAlloca = Builder.CreateAlloca(RuntimeConstantArrayTy);
    // Zero-initialize the alloca to avoid stack garbage for caching.
    Builder.CreateStore(Constant::getNullValue(RuntimeConstantArrayTy),
                        RuntimeConstantsAlloca);
    for (int ArgI = 0; ArgI < JFI.ConstantArgs.size(); ++ArgI) {
      auto *GEP = Builder.CreateInBoundsGEP(
          RuntimeConstantArrayTy, RuntimeConstantsAlloca,
          {Builder.getInt32(0), Builder.getInt32(ArgI)});
      int ArgNo = JFI.ConstantArgs[ArgI];
      Builder.CreateStore(JFI.Fn->getArg(ArgNo), GEP);
    }
    Builder.restoreIP(IP);
  } else
    RuntimeConstantsAlloca =
        Constant::getNullValue(RuntimeConstantArrayTy->getPointerTo());

  assert(RuntimeConstantsAlloca &&
         "Expected non-null runtime constants alloca");

  auto *Ret = Builder.CreateCall(
      JitEntryFn,
      {FnNameGlobal, HipFatbinWrapper, RuntimeConstantsAlloca,
       Builder.getInt32(JFI.ConstantArgs.size()),
       LaunchKernelCall->getArgOperand(1), LaunchKernelCall->getArgOperand(2),
       LaunchKernelCall->getArgOperand(3), LaunchKernelCall->getArgOperand(4),
       LaunchKernelCall->getArgOperand(5), LaunchKernelCall->getArgOperand(6),
       LaunchKernelCall->getArgOperand(7)});

  LaunchKernelCall->replaceAllUsesWith(Ret);
  LaunchKernelCall->eraseFromParent();

  // dbgs() << "=== StubFn " << *JFI.Fn << "=== End of StubFn\n";
}

static void instrumentRegisterVar(Module &M) {
  Function *RegisterVarFn = M.getFunction("__hipRegisterVar");
  if (!RegisterVarFn)
    return;

  // Create jit entry runtime function.
  Type *VoidPtrTy = Type::getInt8PtrTy(M.getContext());

  // The prototype is __jit_register_var(void *HostAddr, const char *VarName).
  FunctionType *JitRegisterVarFnTy =
      FunctionType::get(VoidPtrTy, {VoidPtrTy, VoidPtrTy},
                        /* isVarArg=*/false);
  FunctionCallee JitRegisterVarFn =
      M.getOrInsertFunction("__jit_register_var", JitRegisterVarFnTy);

  for (User *Usr : RegisterVarFn->users())
    if (CallBase *CB = dyn_cast<CallBase>(Usr)) {
      IRBuilder<> Builder(CB->getNextNode());
      Builder.CreateCall(JitRegisterVarFn,
                         {CB->getArgOperand(1), CB->getArgOperand(2)});
    }
}

static void emitJITFunctionStub(Module &M, JitFunctionInfo &JFI) {
  // Create jit entry runtime function.
  Type *VoidPtrTy = Type::getInt8PtrTy(M.getContext());
  Type *Int32Ty = Type::getInt32Ty(M.getContext());
  Type *Int64Ty = Type::getInt64Ty(M.getContext());
  // Use Int64 type for the Value, big enough to hold primitives.
  StructType *RuntimeConstantTy = StructType::create({Int64Ty}, "struct.args");

  FunctionType *JitEntryFnTy =
      FunctionType::get(VoidPtrTy,
                        {VoidPtrTy, VoidPtrTy, Int32Ty,
                         RuntimeConstantTy->getPointerTo(), Int32Ty},
                        /* isVarArg=*/false);
  FunctionCallee JitEntryFn =
      M.getOrInsertFunction("__jit_entry", JitEntryFnTy);

  // Replaces jit'ed functions in the original module with stubs to call the
  // runtime entry point that compiles and links.
  Function *F = JFI.Fn;

  // Replace jit'ed function with a stub function.
  std::string FnName = F->getName().str();
  F->setName("");
  Function *StubFn =
      Function::Create(F->getFunctionType(), F->getLinkage(), FnName, M);
  F->replaceAllUsesWith(StubFn);
  F->eraseFromParent();

  // Replace the body of the jit'ed function to call the jit entry, grab the
  // address of the specialized jit version and execute it.
  IRBuilder<> Builder(BasicBlock::Create(M.getContext(), "entry", StubFn));

  // Create the runtime constant array type for the runtime constants passed
  // to the jit entry function.
  ArrayType *RuntimeConstantArrayTy =
      ArrayType::get(RuntimeConstantTy, JFI.ConstantArgs.size());

  // Create globals for the function name and string IR passed to the jit
  // entry.
  auto *FnNameGlobal = Builder.CreateGlobalString(StubFn->getName());
  auto *StrIRGlobal = Builder.CreateGlobalString(JFI.ModuleIR);

  // Create the runtime constants data structure passed to the jit entry.
  Value *RuntimeConstantsAlloca = nullptr;
  if (JFI.ConstantArgs.size() > 0) {
    // auto *GV = RuntimeConstantsAlloca =
    //     new GlobalVariable(M, RuntimeConstantArrayTy, /* isConstant */
    //     false,
    //                        GlobalValue::InternalLinkage,
    //                        Constant::getNullValue(RuntimeConstantArrayTy),
    //                        StubFn->getName() + ".rconsts");
    RuntimeConstantsAlloca = Builder.CreateAlloca(RuntimeConstantArrayTy);
    // Zero-initialize the alloca to avoid stack garbage for caching.
    Builder.CreateStore(Constant::getNullValue(RuntimeConstantArrayTy),
                        RuntimeConstantsAlloca);
    for (int ArgI = 0; ArgI < JFI.ConstantArgs.size(); ++ArgI) {
      auto *GEP = Builder.CreateInBoundsGEP(
          RuntimeConstantArrayTy, RuntimeConstantsAlloca,
          {Builder.getInt32(0), Builder.getInt32(ArgI)});
      int ArgNo = JFI.ConstantArgs[ArgI];
      Builder.CreateStore(StubFn->getArg(ArgNo), GEP);
    }
  } else
    RuntimeConstantsAlloca =
        Constant::getNullValue(RuntimeConstantArrayTy->getPointerTo());

  assert(RuntimeConstantsAlloca &&
         "Expected non-null runtime constants alloca");

  auto *JitFnPtr = Builder.CreateCall(
      JitEntryFn,
      {FnNameGlobal, StrIRGlobal, Builder.getInt32(JFI.ModuleIR.size()),
       RuntimeConstantsAlloca, Builder.getInt32(JFI.ConstantArgs.size())});
  SmallVector<Value *, 8> Args;
  for (auto &Arg : StubFn->args())
    Args.push_back(&Arg);
  auto *RetVal = Builder.CreateCall(StubFn->getFunctionType(), JitFnPtr, Args);
  if (StubFn->getReturnType()->isVoidTy())
    Builder.CreateRetVoid();
  else
    Builder.CreateRet(RetVal);

  // dbgs() << "=== StubFn " << *StubFn << "=== End of StubFn\n";
}

// This method implements what the pass does
void visitor(Module &M, CallGraph &CG) {

  if (JitFunctionInfoList.empty()) {
    // dbgs() << "=== Empty Begin Mod\n" << M << "=== End Mod\n";
    return;
  }

  DEBUG(dbgs() << "=== Pre M\n" << M << "=== End of Pre M\n");
  // TODO: Supporting -fgpu-rdc compilation is tricky because clang compiles
  // first for the host and then for the device. The method here will not work.
  // It assumes that device compilation runs first to output JIT section
  // bitcodes that the host compilation ingests. One possibility is to let the
  // runtime library extract the JIT section bitcodes to perform preprocessing
  // and jit compilation.
  if (isDeviceCompilation(M)) {
    for (JitFunctionInfo &JFI : JitFunctionInfoList)
      createJITModuleSectionsDevice(M, JFI);

    return;
  }

  if (HasDeviceKernels(M))
    instrumentRegisterVar(M);

  // First pass creates the string Module IR per jit'ed function.
  for (JitFunctionInfo &JFI : JitFunctionInfoList) {
    dbgs() << "Processing JIT Function " << JFI.Fn->getName() << "\n";
    Value *Kernel = nullptr;
    CallBase *LaunchKernelCall = nullptr;
    if (isDeviceKernel(M, *JFI.Fn, &LaunchKernelCall, &Kernel)) {
      // TODO: Assumes Kernel->getName() returns the actual string name of the
      // kernel, a more robust solution would look up hipRegisterFunction to
      // find the name by the argument global.
      assert(Kernel != nullptr && "Expected non-null kernel");

      emitJITKernelStub(M, JFI, Kernel->getName(), LaunchKernelCall);
      dbgs() << "DONE!\n";

      continue;
    }

    createJITModule(M, JFI);
    emitJITFunctionStub(M, JFI);
  }

  DEBUG(dbgs() << "=== Post M\n" << M << "=== End Post M\n");
  if (verifyModule(M, &errs()))
    report_fatal_error("Broken original module found, compilation aborted!",
                       false);
  else
    dbgs() << "Module verified!\n";
}

// New PM implementation
struct JitPass : PassInfoMixin<JitPass> {
  // Main entry point, takes IR unit to run the pass on (&F) and the
  // corresponding pass manager (to be queried if need be)
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {
    parseAnnotations(M);
    CallGraph &CG = AM.getResult<CallGraphAnalysis>(M);
    visitor(M, CG);
    // TODO: is anything preserved?
    return PreservedAnalyses::none();
    // return PreservedAnalyses::all();
  }

  // Without isRequired returning true, this pass will be skipped for functions
  // decorated with the optnone LLVM attribute. Note that clang -O0 decorates
  // all functions with optnone.
  static bool isRequired() { return true; }
};

// Legacy PM implementation
struct LegacyJitPass : public ModulePass {
  static char ID;
  LegacyJitPass() : ModulePass(ID) {}
  // Main entry point - the name conveys what unit of IR this is to be run on.
  bool runOnModule(Module &M) override {
    parseAnnotations(M);
    CallGraph &CG = getAnalysis<CallGraphWrapperPass>().getCallGraph();
    visitor(M, CG);

    // TODO: what is preserved?
    return true;
    // Doesn't modify the input unit of IR, hence 'false'
    // return false;
  }
};
} // namespace

//-----------------------------------------------------------------------------
// New PM Registration
//-----------------------------------------------------------------------------
llvm::PassPluginLibraryInfo getJitPassPluginInfo() {
  const auto callback = [](PassBuilder &PB) {
  // TODO: decide where to insert it in the pipeline. Early avoids
  // inlining jit function (which disables jit'ing) but may require more
  // optimization, hence overhead, at runtime.
  // PB.registerPipelineStartEPCallback([&](ModulePassManager &MPM, auto) {
#if ENABLE_HIP
    // NOTE:: For device jitting register the pass late, reduces compilation
    // time and does not have risks of losing the kernel due to inlining.
    // PB.registerPipelineStartEPCallback([&](ModulePassManager &MPM, auto)
    // {
    // NOTE: Investigating why late pass execution to extract bc on device fails
    // to link in the runtime with error:
    // ":1:hiprtcInternal.cpp:654 : 741216360548 us: [pid:3847761
    // tid:0x15553e8b9ac0] Error in hiprtc: unable to add device libs to
    // linked bitcode
    //: 3:hiprtc.cpp               :364 : 741216360556 us: [pid:3847761
    //: tid:0x15553e8b9ac0] hiprtcLinkComplete: Returned
    //: HIPRTC_ERROR_LINKING
    // ERROR @
    // /p/vast1/ggeorgak/projects/compilers/jitproject/jit/lib/jit.cpp:877
    // -> HIPRTC_ERROR_LINKING"
    // Working assumption is that the hiprtc linker cannot re-link already
    // linked device libraries and aborts.
    PB.registerPipelineEarlySimplificationEPCallback(
        [&](ModulePassManager &MPM, auto) {
    // PB.registerOptimizerLastEPCallback([&](ModulePassManager &MPM, auto) {
#else
    // NOTE: For host jitting register the pass earlier.
    PB.registerPipelineEarlySimplificationEPCallback(
        [&](ModulePassManager &MPM, auto) {
#endif
          MPM.addPass(JitPass());
          return true;
        });
  };

  return {LLVM_PLUGIN_API_VERSION, "JitPass", LLVM_VERSION_STRING, callback};
}

// TODO: use by jit-pass name.
// This is the core interface for pass plugins. It guarantees that 'opt' will
// be able to recognize JitPass when added to the pass pipeline on the
// command line, i.e. via '-passes=jit-pass'
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getJitPassPluginInfo();
}

//-----------------------------------------------------------------------------
// Legacy PM Registration
//-----------------------------------------------------------------------------
// The address of this variable is used to uniquely identify the pass. The
// actual value doesn't matter.
char LegacyJitPass::ID = 0;

// This is the core interface for pass plugins. It guarantees that 'opt' will
// recognize LegacyJitPass when added to the pass pipeline on the command
// line, i.e.  via '--legacy-jit-pass'
static RegisterPass<LegacyJitPass>
    X("legacy-jit-pass", "Jit Pass",
      false, // This pass doesn't modify the CFG => false
      false  // This pass is not a pure analysis pass => false
    );
