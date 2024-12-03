//===-- ProteusJitPass.cpp -- Extact code/runtime info for Proteus JIT --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
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
//      opt -enable-new-pm=0 -load libProteusJitPass.dylib -legacy-jit-pass
//      -disable-output `\`
//        <input-llvm-file>
//    2. New PM
//      opt -load-pass-plugin=libProteusJitPass.dylib -passes="jit-pass" `\`
//        -disable-output <input-llvm-file>
//
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/CallGraph.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Object/ELF.h"
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
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/CallingConv.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/MemoryBufferRef.h>

#include <iostream>
#include <string>

#define DEBUG_TYPE "jitpass"
#ifdef ENABLE_DEBUG
#define DEBUG(x) x
#else
#define DEBUG(x)
#endif

#define FATAL_ERROR(x)                                                         \
  report_fatal_error(llvm::Twine(std::string{} + __FILE__ + ":" +              \
                                 std::to_string(__LINE__) + " => " + x))

#if ENABLE_HIP
constexpr char const *RegisterFunctionName = "__hipRegisterFunction";
constexpr char const *LaunchFunctionName = "hipLaunchKernel";
constexpr char const *FatbinName = "__hip_fatbin_wrapper";
constexpr char const *RegisterVarName = "__hipRegisterVar";
#elif ENABLE_CUDA
constexpr char const *RegisterFunctionName = "__cudaRegisterFunction";
constexpr char const *LaunchFunctionName = "cudaLaunchKernel";
constexpr char const *FatbinName = "__cuda_fatbin_wrapper";
constexpr char const *RegisterVarName = "__cudaRegisterVar";
#else
constexpr char const *RegisterFunctionName = nullptr;
constexpr char const *LaunchFunctionName = nullptr;
constexpr char const *FatbinName = nullptr;
constexpr char const *RegisterVarName = nullptr;
#endif

using namespace llvm;

//-----------------------------------------------------------------------------
// ProteusJitPass implementation
//-----------------------------------------------------------------------------
namespace {

struct ProteusJitPassImpl {
  struct JitFunctionInfo {
    SmallVector<int, 8> ConstantArgs;
    std::string ModuleIR;
  };

  MapVector<Function *, JitFunctionInfo> JitFunctionInfoMap;
  DenseMap<Value *, GlobalVariable *> StubToKernelMap;
  SmallPtrSet<Function *, 16> ModuleDeviceKernels;

  bool isDeviceCompilation(Module &M) {
    Triple TargetTriple(M.getTargetTriple());
    DEBUG(dbgs() << "TargetTriple " << M.getTargetTriple() << "\n");
    if (TargetTriple.isNVPTX() || TargetTriple.isAMDGCN())
      return true;

    return false;
  }

  bool isDeviceKernel(const Function *F) {
    if (ModuleDeviceKernels.contains(F))
      return true;

    return false;
  }

  void parseAnnotations(Module &M) {
    auto GlobalAnnotations = M.getNamedGlobal("llvm.global.annotations");
    if (!GlobalAnnotations)
      return;

    auto Array = cast<ConstantArray>(GlobalAnnotations->getOperand(0));
    DEBUG(dbgs() << "Annotation Array " << *Array << "\n");
    for (int i = 0; i < Array->getNumOperands(); i++) {
      auto Entry = cast<ConstantStruct>(Array->getOperand(i));
      DEBUG(dbgs() << "Entry " << *Entry << "\n");

      auto Fn = dyn_cast<Function>(Entry->getOperand(0)->stripPointerCasts());

      assert(Fn && "Expected function in entry operands");

      // Check the annotated functions is a kernel function.
      if (isDeviceCompilation(M)) {
        ModuleDeviceKernels = getDeviceKernels(M);
        if (!isDeviceKernel(Fn))
          FATAL_ERROR(std::string{} + __FILE__ + ":" +
                      std::to_string(__LINE__) +
                      " => Expected the annotated Fn " + Fn->getName() +
                      " to be a kernel function!");
      }

      if (JitFunctionInfoMap.contains(Fn))
        FATAL_ERROR("Duplicate jit annotation for Fn " + Fn->getName());

      DEBUG(dbgs() << "JIT Function " << Fn->getName() << "\n");

      auto Annotation =
          cast<ConstantDataArray>(Entry->getOperand(1)->getOperand(0));

      DEBUG(dbgs() << "Annotation " << Annotation->getAsCString() << "\n");

      // TODO: needs CString for comparison to work, why?
      if (Annotation->getAsCString().compare("jit"))
        continue;

      JitFunctionInfo JFI;

      if (Entry->getOperand(4)->isNullValue())
        JFI.ConstantArgs = {};
      else {
        DEBUG(dbgs() << "AnnotArgs " << *Entry->getOperand(4)->getOperand(0)
                     << "\n");
        DEBUG(dbgs() << "Type AnnotArgs "
                     << *Entry->getOperand(4)->getOperand(0)->getType()
                     << "\n");
        auto AnnotArgs =
            cast<ConstantStruct>(Entry->getOperand(4)->getOperand(0));

        for (int I = 0; I < AnnotArgs->getNumOperands(); ++I) {
          auto *Index = cast<ConstantInt>(AnnotArgs->getOperand(I));
          uint64_t ArgNo = Index->getValue().getZExtValue();
          if (ArgNo > Fn->arg_size())
            FATAL_ERROR(
                Twine("Error: JIT annotation runtime constant argument " +
                      std::to_string(ArgNo) +
                      " is greater than number of arguments " +
                      std::to_string(Fn->arg_size()))
                    .str()
                    .c_str());
          // TODO: think about types, -1 to convert to 0-start index.
          JFI.ConstantArgs.push_back(ArgNo - 1);
        }
      }

      JitFunctionInfoMap[Fn] = JFI;
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
    Passes.addPass(GlobalDCEPass());
    Passes.addPass(StripDeadDebugInfoPass());
    Passes.addPass(StripDeadPrototypesPass());

    Passes.run(M, MAM);
  }

  void runOptimizationPassPipeline(Module &M) {
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

    ModulePassManager Passes =
        PB.buildPerModuleDefaultPipeline(OptimizationLevel::O3);
    Passes.run(M, MAM);
  }

  void emitJitModuleHost(Module &M,
                         std::pair<Function *, JitFunctionInfo> &JITInfo) {
    Function *JITFn = JITInfo.first;
    JitFunctionInfo &JFI = JITInfo.second;

    ValueToValueMapTy VMap;
    auto JitMod = CloneModule(M, VMap, [](const GlobalValue *GV) {
      if (const GlobalVariable *G = dyn_cast<GlobalVariable>(GV))
        if (!G->isConstant())
          return false;

      return true;
    });

    Function *JitF = cast<Function>(VMap[JITFn]);
    JitF->setLinkage(GlobalValue::ExternalLinkage);

    // Internalize functions, besides JIT function, in the module
    // to help global DCE (reduce compilation time), inlining.
    for (Function &JitModF : *JitMod) {
      if (JitModF.isDeclaration())
        continue;

      if (&JitModF == JitF)
        continue;

      // Internalize other functions in the module.
      JitModF.setLinkage(GlobalValue::InternalLinkage);
    }

    DEBUG(dbgs() << "=== Pre Passes Host JIT Module\n"
                 << *JitMod << "=== End of Pre Passes Host JIT Module\n");

    // Run a global DCE pass and O3 on the JIT module IR to remove unnecessary
    // symbols and reduce the IR to JIT at runtime.
    runCleanupPassPipeline(*JitMod);
    runOptimizationPassPipeline(*JitMod);

    // Update linkage and visibility in the original module only for
    // globals included in the JIT module required for external
    // linking.
    for (auto &GVar : M.globals()) {
      auto printGVarInfo = [](auto &GVar) {
        dbgs() << "=== GVar\n";
        dbgs() << GVar.getName() << "\n";
        dbgs() << "Linkage " << GVar.getLinkage() << "\n";
        dbgs() << "Visibility " << GVar.getVisibility() << "\n";
        dbgs() << "=== End GV\n";
      };

      if (VMap[&GVar]) {
        DEBUG(printGVarInfo(GVar));

        if (GVar.isConstant())
          continue;

        if (GVar.getName() == "llvm.global_ctors") {
          DEBUG(dbgs() << "Skip llvm.global_ctors");
          continue;
        }

        if (GVar.hasAvailableExternallyLinkage()) {
          DEBUG(dbgs() << "Skip available externally");
          continue;
        }

        GVar.setLinkage(GlobalValue::ExternalLinkage);
        GVar.setVisibility(GlobalValue::VisibilityTypes::DefaultVisibility);
      }
    }

    // TODO: Do we want to keep debug info to use for GDB/LLDB
    // interfaces for debugging jitted code?
    StripDebugInfo(*JitMod);

    // Add metadata for the JIT function to store the argument positions for
    // runtime constants.
    emitJitFunctionArgMetadata(*JitMod, JFI, *JitF);

    if (verifyModule(*JitMod, &errs()))
      FATAL_ERROR("Broken JIT module found, compilation aborted!");

    raw_string_ostream OS(JFI.ModuleIR);
    WriteBitcodeToFile(*JitMod, OS);
    OS.flush();

    DEBUG(dbgs() << "=== Final Host JIT Module\n"
                 << *JitMod << "=== End of Final Host JIT Module\n");
  }

  void emitJitModuleDevice(Module &M,
                           std::pair<Function *, JitFunctionInfo> &JITInfo) {
    ValueToValueMapTy VMap;
    Function *JITFn = JITInfo.first;
    JitFunctionInfo &JFI = JITInfo.second;

    auto JitMod = CloneModule(M, VMap, [&](const GlobalValue *GV) {
      // Do not clone JIT bitcodes of other kernels, assumes the existing
      // special naming for globals storing the bitcodes.
      if (GV->getName().starts_with("__jit_bc"))
        return false;

      if (const Function *F = dyn_cast<Function>(GV)) {
        if (F == JITFn)
          return true;
        // Do not clone other host-callable kernels.
        // TODO: Is this necessary when using hiprtc? In any case it reduces
        // the JITted code which should reduce compilation time.
        if (isDeviceKernel(F))
          return false;
      }

      return true;
    });

    // Remove llvm.global.annotations and .jit section globals from the module
    // and used lists.
    SmallPtrSet<GlobalVariable *, 8> JitGlobalsToRemove;
    auto JitGlobalAnnotations =
        JitMod->getNamedGlobal("llvm.global.annotations");
    assert(JitGlobalAnnotations &&
           "Expected llvm.global.annotations in jit module");
    JitGlobalsToRemove.insert(JitGlobalAnnotations);

    // Clang emits this global variable in RDC compilation. We need to remove it
    // to avoid dangling references to unused kernels that are removed.
    auto *JitClangGPUUsedExternals =
        JitMod->getNamedGlobal("__clang_gpu_used_external");
    if (JitClangGPUUsedExternals)
      JitGlobalsToRemove.insert(JitClangGPUUsedExternals);

    for (auto &GV : JitMod->globals()) {
      if (GV.getName().starts_with("__jit_bc"))
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
      GV->eraseFromParent();

    ModuleDeviceKernels = getDeviceKernels(*JitMod);
    SmallPtrSet<Function *, 8> JitUnusedKernelsToRemove;
    for (auto &JitModF : *JitMod) {
      // Remove unused kernels that have been demoted to declarations during
      // cloning.
      if (JitModF.isDeclaration())
        if (isDeviceKernel(&JitModF))
          JitUnusedKernelsToRemove.insert(&JitModF);
    }

    for (auto *F : JitUnusedKernelsToRemove)
      F->eraseFromParent();

    Function *JitF = cast<Function>(VMap[JITFn]);

    // Internalize functions, besides JIT function, in the module
    // to help global DCE (reduce compilation time), inlining.
    for (Function &JitModF : *JitMod) {
      if (JitModF.isDeclaration())
        continue;

      if (&JitModF == JitF)
        continue;

      // Internalize other functions in the module.
      JitModF.setLinkage(GlobalValue::InternalLinkage);
    }

    DEBUG(dbgs() << "=== Pre Passes Device JIT Module\n"
                 << *JitMod << "=== End of Pre Passes Device JIT Module\n");

    // Run a global DCE pass on the JIT module IR to remove
    // unnecessary symbols and reduce the IR to JIT at runtime.
    runCleanupPassPipeline(*JitMod);

#if ENABLE_DEBUG
    {
      // NOTE: We must have a single kernel per JIT module, otherwise CUDA/HIP
      // RTC interfaces fail with linking errors.
      auto JITKernels = getDeviceKernels(*JitMod);
      if (JITKernels.size() != 1)
        FATAL_ERROR("Expected a single kernel in JIT module");
    }

#endif
    // TODO: Decide whether to run optimization on the extracted module.
    // runOptimizationPassPipeline(*JitMod);

    DEBUG(dbgs() << "=== Post Passes Device JIT Module\n"
                 << *JitMod << "=== End of Post Passes Device JIT Module\n");

    // TODO: Do we want to keep debug info to use for GDB/LLDB
    // interfaces for debugging jitted code?
    StripDebugInfo(*JitMod);

    // Add metadata for the JIT function to store the argument positions for
    // runtime constants.
    emitJitFunctionArgMetadata(*JitMod, JFI, *JitF);

    if (verifyModule(*JitMod, &errs()))
      FATAL_ERROR("Broken JIT module found, compilation aborted!");

    raw_string_ostream OS(JFI.ModuleIR);
    WriteBitcodeToFile(*JitMod, OS);
    OS.flush();

    // NOTE: HIP compilation supports custom section in the binary to store the
    // IR. CUDA does not, hence we parse the IR by reading the global from the
    // device memory.
    Constant *JitModule = ConstantDataArray::get(
        M.getContext(), ArrayRef<uint8_t>((const uint8_t *)JFI.ModuleIR.data(),
                                          JFI.ModuleIR.size()));
    auto *GV = new GlobalVariable(
        M, JitModule->getType(), /* isConstant */ true,
        GlobalValue::PrivateLinkage, JitModule, "__jit_bc_" + JITFn->getName());
    appendToUsed(M, {GV});
#if ENABLE_HIP
    GV->setSection(Twine(".jit." + JITFn->getName()).str());
#endif

    DEBUG(dbgs() << "=== Post Device Original Module\n"
                 << M << "=== End Post Device Original Module M\n");

    return;
  }

  void emitJitFunctionArgMetadata(Module &JitMod, JitFunctionInfo &JFI,
                                  Function &JitF) {
    LLVMContext &Ctx = JitMod.getContext();
    SmallVector<Metadata *> ConstArgNos;
    for (size_t I = 0; I < JFI.ConstantArgs.size(); ++I) {
      int ArgNo = JFI.ConstantArgs[I];
      Metadata *Meta = ConstantAsMetadata::get(
          ConstantInt::get(Type::getInt32Ty(Ctx), ArgNo));
      ConstArgNos.push_back(Meta);
    }
    MDNode *Node = MDNode::get(Ctx, ConstArgNos);
    JitF.setMetadata("jit_arg_nos", Node);
  }

  GlobalVariable *emitModuleUniqueIdGlobal(Module &M) {
    Constant *ModuleUniqueId =
        ConstantDataArray::getString(M.getContext(), getUniqueModuleId(&M));
    auto *GV = new GlobalVariable(M, ModuleUniqueId->getType(), true,
                                  GlobalValue::PrivateLinkage, ModuleUniqueId,
                                  "__module_unique_id");
    appendToUsed(M, {GV});

    return GV;
  }

  void emitJitEntryCall(Module &M,
                        std::pair<Function *, JitFunctionInfo> &JITInfo) {

    Function *JITFn = JITInfo.first;
    JitFunctionInfo &JFI = JITInfo.second;

    // Create jit entry runtime function.
    Type *VoidPtrTy = PointerType::get(M.getContext(), 0);
    Type *Int32Ty = Type::getInt32Ty(M.getContext());
    Type *Int128Ty = Type::getInt128Ty(M.getContext());
    // Use Int64 type for the Value, big enough to hold primitives.
    StructType *RuntimeConstantTy =
        StructType::create({Int128Ty}, "struct.args");

    FunctionType *JitEntryFnTy =
        FunctionType::get(VoidPtrTy,
                          {VoidPtrTy, VoidPtrTy, Int32Ty,
                           RuntimeConstantTy->getPointerTo(), Int32Ty},
                          /* isVarArg=*/false);
    FunctionCallee JitEntryFn =
        M.getOrInsertFunction("__jit_entry", JitEntryFnTy);

    // Replaces jit'ed functions in the original module with stubs to call the
    // runtime entry point that compiles and links.
    // Replace jit'ed function with a stub function.
    std::string FnName = JITFn->getName().str();
    JITFn->setName("");
    Function *StubFn = Function::Create(JITFn->getFunctionType(),
                                        JITFn->getLinkage(), FnName, M);
    JITFn->replaceAllUsesWith(StubFn);
    JITFn->eraseFromParent();

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
    Value *RuntimeConstantsIndicesAlloca = nullptr;
    if (JFI.ConstantArgs.size() > 0) {
      RuntimeConstantsIndicesAlloca =
          Builder.CreateAlloca(RuntimeConstantArrayTy);
      // Zero-initialize the alloca to avoid stack garbage for caching.
      Builder.CreateStore(Constant::getNullValue(RuntimeConstantArrayTy),
                          RuntimeConstantsIndicesAlloca);
      for (int ArgI = 0; ArgI < JFI.ConstantArgs.size(); ++ArgI) {
        auto *GEP = Builder.CreateInBoundsGEP(
            RuntimeConstantArrayTy, RuntimeConstantsIndicesAlloca,
            {Builder.getInt32(0), Builder.getInt32(ArgI)});
        int ArgNo = JFI.ConstantArgs[ArgI];
        Builder.CreateStore(StubFn->getArg(ArgNo), GEP);
      }
    } else
      RuntimeConstantsIndicesAlloca =
          Constant::getNullValue(RuntimeConstantArrayTy->getPointerTo());

    assert(RuntimeConstantsIndicesAlloca &&
           "Expected non-null runtime constants alloca");

    auto *JitFnPtr = Builder.CreateCall(
        JitEntryFn,
        {FnNameGlobal, StrIRGlobal, Builder.getInt32(JFI.ModuleIR.size()),
         RuntimeConstantsIndicesAlloca,
         Builder.getInt32(JFI.ConstantArgs.size())});
    SmallVector<Value *, 8> Args;
    for (auto &Arg : StubFn->args())
      Args.push_back(&Arg);
    auto *RetVal =
        Builder.CreateCall(StubFn->getFunctionType(), JitFnPtr, Args);
    if (StubFn->getReturnType()->isVoidTy())
      Builder.CreateRetVoid();
    else
      Builder.CreateRet(RetVal);
  }

  Value *getStubGV(Value *Operand) {
    // NOTE: when called by isDeviceKernelHostStub, Operand may not be a global
    // variable point to the stub, so we check and return null in that case.
    Value *V = nullptr;
#if ENABLE_HIP
    // NOTE: Hip creates a global named after the device kernel function that
    // points to the host kernel stub. Because of this, we need to unpeel this
    // indirection to use the host kernel stub for finding the device kernel
    // function name global.
    GlobalVariable *IndirectGV = dyn_cast<GlobalVariable>(Operand);
    V = IndirectGV ? IndirectGV->getInitializer() : nullptr;
#elif ENABLE_CUDA
    GlobalValue *DirectGV = dyn_cast<GlobalValue>(Operand);
    V = DirectGV ? DirectGV : nullptr;
#endif

    return V;
  }

  void getKernelHostStubs(Module &M) {
    Function *RegisterFunction = nullptr;
    if (!RegisterFunctionName) {
      FATAL_ERROR("getKernelHostStubs only callable with `EnableHIP or "
                  "EnableCUDA set.");
      return;
    }
    RegisterFunction = M.getFunction(RegisterFunctionName);

    if (!RegisterFunction)
      return;

    constexpr int StubOperand = 1;
    constexpr int KernelOperand = 2;
    for (User *Usr : RegisterFunction->users())
      if (CallBase *CB = dyn_cast<CallBase>(Usr)) {
        GlobalVariable *GV =
            dyn_cast<GlobalVariable>(CB->getArgOperand(KernelOperand));
        assert(GV && "Expected global variable as kernel name operand");
        Value *Key = getStubGV(CB->getArgOperand(StubOperand));
        assert(Key && "Expected valid kernel stub key");
        StubToKernelMap[Key] = GV;
        DEBUG(dbgs() << "StubToKernelMap Key: " << Key->getName() << " -> "
                     << *GV << "\n");
      }
  }

  SmallPtrSet<Function *, 16> getDeviceKernels(Module &M) {
    SmallPtrSet<Function *, 16> Kernels;
#if ENABLE_CUDA
    NamedMDNode *MD = M.getOrInsertNamedMetadata("nvvm.annotations");

    if (!MD)
      return Kernels;

    for (auto *Op : MD->operands()) {
      if (Op->getNumOperands() < 2)
        continue;
      MDString *KindID = dyn_cast<MDString>(Op->getOperand(1));
      if (!KindID || KindID->getString() != "kernel")
        continue;

      Function *KernelFn =
          mdconst::dyn_extract_or_null<Function>(Op->getOperand(0));
      if (!KernelFn)
        continue;

      Kernels.insert(KernelFn);
    }
#elif ENABLE_HIP
    for (Function &F : M)
      if (F.getCallingConv() == CallingConv::AMDGPU_KERNEL)
        Kernels.insert(&F);
#endif

    return Kernels;
  }

  bool isDeviceKernelHostStub(Module &M, Function &Fn) {
    if (StubToKernelMap.contains(&Fn))
      return true;

    return false;
  }

  bool hasDeviceLaunchKernelCalls(Module &M) {
    Function *LaunchKernelFn = nullptr;
    if (!LaunchFunctionName) {
      return false;
    }
    LaunchKernelFn = M.getFunction(LaunchFunctionName);

    if (!LaunchKernelFn)
      return false;

    return true;
  }

  void replaceWithJitLaunchKernel(Module &M, CallBase *LaunchKernelCall) {
    // Create jit entry runtime function.
    Type *VoidPtrTy = PointerType::get(M.getContext(), 0);
    Type *Int64Ty = Type::getInt64Ty(M.getContext());
    Type *Int32Ty = Type::getInt32Ty(M.getContext());

    GlobalVariable *ModuleUniqueId =
        M.getGlobalVariable("__module_unique_id", true);
    assert(ModuleUniqueId && "Expected ModuleUniqueId global to be defined");

    if (!FatbinName)
      FATAL_ERROR("replaceWithJitLaunchKernel not callable without EnableHIP "
                  "or EnableCUDA set.");

    GlobalVariable *FatbinWrapper = nullptr;
    FatbinWrapper = M.getGlobalVariable(FatbinName, true);
    if (!FatbinWrapper)
      FATAL_ERROR(
          "Expected hip fatbinary wrapper global, check "
          "ENABLE_CUDA|ENABLE_HIP compilation flags for ProteusJitPass");

    FunctionType *JitEntryFnTy = nullptr;
#if ENABLE_HIP
    JitEntryFnTy =
        FunctionType::get(Int32Ty,
                          {ModuleUniqueId->getType(), VoidPtrTy,
                           FatbinWrapper->getType(), Int64Ty, Int64Ty, Int32Ty,
                           Int64Ty, Int32Ty, VoidPtrTy, Int64Ty, VoidPtrTy},
                          /* isVarArg=*/false);
#elif ENABLE_CUDA
    // NOTE: CUDA uses an array type for passing grid, block sizes.
    JitEntryFnTy =
        FunctionType::get(Int32Ty,
                          {ModuleUniqueId->getType(),
                           VoidPtrTy, // Kernel address
                           FatbinWrapper->getType(),
                           Int64Ty,                    // Fatbin size
                           ArrayType::get(Int64Ty, 2), // Grid dim array
                           ArrayType::get(Int64Ty, 2), // Block dim array
                           VoidPtrTy,                  // Kernel args
                           Int64Ty,                    // Shared mem size
                           VoidPtrTy},
                          /* isVarArg=*/false);
#endif

    if (!JitEntryFnTy)
      FATAL_ERROR(
          "Expected non-null jit entry function type, check "
          "ENABLE_CUDA|ENABLE_HIP compilation flags for ProteusJitPass");

    FunctionCallee JitEntryFn =
        M.getOrInsertFunction("__jit_launch_kernel", JitEntryFnTy);

    // Insert before the launch kernel call instruction.
    IRBuilder<> Builder(LaunchKernelCall);
    CallInst *Call = nullptr;
#ifdef ENABLE_HIP
    Call = Builder.CreateCall(
        JitEntryFn,
        {ModuleUniqueId, LaunchKernelCall->getArgOperand(0), FatbinWrapper,
         /* FatbinSize unused by HIP */ Builder.getInt64(0),
         LaunchKernelCall->getArgOperand(1), LaunchKernelCall->getArgOperand(2),
         LaunchKernelCall->getArgOperand(3), LaunchKernelCall->getArgOperand(4),
         LaunchKernelCall->getArgOperand(5), LaunchKernelCall->getArgOperand(6),
         LaunchKernelCall->getArgOperand(7)});
#elif ENABLE_CUDA
    ConstantStruct *C =
        dyn_cast<ConstantStruct>(FatbinWrapper->getInitializer());
    assert(C->getType()->getNumElements() &&
           "Expected four fields in fatbin wrapper struct");
    constexpr int FatbinField = 2;
    auto *Fatbin = C->getAggregateElement(FatbinField);
    GlobalVariable *FatbinGV = dyn_cast<GlobalVariable>(Fatbin);
    assert(FatbinGV && "Expected global variable for the fatbin object");
    ArrayType *ArrayTy =
        dyn_cast<ArrayType>(FatbinGV->getInitializer()->getType());
    assert(ArrayTy && "Expected array type of the fatbin object");
    size_t FatbinSize = ArrayTy->getNumElements();

    Call = Builder.CreateCall(
        JitEntryFn, {
                        ModuleUniqueId,
                        LaunchKernelCall->getArgOperand(0), // Kernel address
                        FatbinWrapper, Builder.getInt64(FatbinSize),
                        LaunchKernelCall->getArgOperand(1), // Grid dim
                        LaunchKernelCall->getArgOperand(2), // Block dim
                        LaunchKernelCall->getArgOperand(3), // Kernel args
                        LaunchKernelCall->getArgOperand(4), // Shmem size
                        LaunchKernelCall->getArgOperand(5)  // Stream
                    });
#endif

    if (!Call)
      FATAL_ERROR(
          "Expected non-null jit launch kernel call, check "
          "ENABLE_CUDA|ENABLE_HIP compilation flags for ProteusJitPass");

    LaunchKernelCall->replaceAllUsesWith(Call);
    LaunchKernelCall->eraseFromParent();
  }

  void
  emitJitLaunchKernelCall(Module &M,
                          std::pair<Function *, JitFunctionInfo> &JITInfo) {
    Function *LaunchKernelFn = nullptr;
    if (!LaunchFunctionName) {
      FATAL_ERROR(
          "Expected non-null LaunchKernelFn, check "
          "ENABLE_CUDA|ENABLE_HIP compilation flags for ProteusJitPass");
    }
    LaunchKernelFn = M.getFunction(LaunchFunctionName);
    if (!LaunchKernelFn)
      FATAL_ERROR(
          "Expected non-null LaunchKernelFn, check "
          "ENABLE_CUDA|ENABLE_HIP compilation flags for ProteusJitPass");

    SmallVector<CallBase *> ToBeReplaced;
    for (User *Usr : LaunchKernelFn->users())
      if (CallBase *CB = dyn_cast<CallBase>(Usr)) {
        // NOTE: We search for calls to the LaunchKernelFn that directly call
        // the kernel through its global value to replace with JIT kernel
        // entries. For cudaLaunchKernel first operand is the stub function,
        // whereas for hipLaunchKernel it is a global variable that points to
        // the stub function. Hence we use GlobalValue instead of
        // GlobalVaraible.
        // TODO: Instrument for indirect launching.

        ToBeReplaced.push_back(CB);
      }

    for (CallBase *CB : ToBeReplaced)
      replaceWithJitLaunchKernel(M, CB);
  }

  void instrumentRegisterVar(Module &M) {
    Function *RegisterVarFn = nullptr;
    if (!RegisterVarName) {
      FATAL_ERROR("instrumentRegisterVar is not callable without EnableCUDA or "
                  "EnableHIP set.");
    }
    RegisterVarFn = M.getFunction(RegisterVarName);
    if (!RegisterVarFn)
      return;


    Type *VoidPtrTy = PointerType::get(M.getContext(), 0);

    // The prototype is __jit_register_var(const void *HostAddr, const char
    // *VarName).
    FunctionType *JitRegisterVarFnTy =
        FunctionType::get(VoidPtrTy, {VoidPtrTy, VoidPtrTy},
                          /* isVarArg=*/false);
    FunctionCallee JitRegisterVarFn =
        M.getOrInsertFunction("__jit_register_var", JitRegisterVarFnTy);

    for (User *Usr : RegisterVarFn->users())
      if (CallBase *CB = dyn_cast<CallBase>(Usr)) {
        IRBuilder<> Builder(CB->getNextNode());
        Value *Symbol = CB->getArgOperand(1);
        auto *GV = dyn_cast<GlobalVariable>(Symbol);
        // GV->setDSOLocal(false);
        // GV->setVisibility(llvm::GlobalValue::DefaultVisibility);
        Value *SymbolName = CB->getArgOperand(2);
        Builder.CreateCall(JitRegisterVarFn, {GV, SymbolName});
      }
  }

  /// instrumentRegisterJITFunc instruments any function passed to a kernel
  /// launch in the IR with a `__jit_register_function` call.
  void instrumentRegisterJITFunc(Module &M) {
    if (!RegisterFunctionName) {
      FATAL_ERROR("instrumentRegisterJITFunc only callable with `EnableHIP or "
                  "EnableCUDA set.");
      return;
    }
    Function *RegisterFunction = M.getFunction(RegisterFunctionName);
    assert(RegisterFunction &&
           "Expected register function to be called at least once.");

    for (User *RegisterFunctionUser : RegisterFunction->users()) {
      CallBase *RegisterCB = dyn_cast<CallBase>(RegisterFunctionUser);
      if (!RegisterCB)
        continue;

      Function *FunctionToRegister =
          dyn_cast<Function>(getStubGV(RegisterCB->getArgOperand(1)));
      assert(FunctionToRegister &&
             "Expected function passed to register function call");
      if (!JitFunctionInfoMap.contains(FunctionToRegister)) {
        DEBUG(dbgs() << "Not instrumenting device kernel "
                     << *FunctionToRegister << "\n");
        continue;
      }

      DEBUG(dbgs() << "Instrumenting JIT function " << *FunctionToRegister
                   << "\n");
      const auto &JFI = JitFunctionInfoMap[FunctionToRegister];
      size_t NumRuntimeConstants = JFI.ConstantArgs.size();
      // Create jit entry runtime function.
      Type *VoidPtrType = Type::getInt8PtrTy(M.getContext());
      Type *VoidType = Type::getVoidTy(M.getContext());
      Type *Int32PtrType = Type::getInt32PtrTy(M.getContext());
      Type *Int32Type = Type::getInt32Ty(M.getContext());

      ArrayType *RuntimeConstantIdxArrayTy =
          ArrayType::get(Int32Type, NumRuntimeConstants);

      IRBuilder<> Builder(RegisterCB->getNextNode());
      // Create an array representing the indices of the args which are runtime
      // constants.
      Value *RuntimeConstantsIndicesAlloca = nullptr;
      RuntimeConstantsIndicesAlloca =
          Builder.CreateAlloca(RuntimeConstantIdxArrayTy);
      assert(RuntimeConstantsIndicesAlloca &&
             "Expected non-null runtime constants alloca");
      // Zero-initialize the alloca to avoid stack garbage for caching.
      Builder.CreateStore(Constant::getNullValue(RuntimeConstantIdxArrayTy),
                          RuntimeConstantsIndicesAlloca);

      for (int ArgI = 0; ArgI < NumRuntimeConstants; ++ArgI) {
        auto *GEP = Builder.CreateInBoundsGEP(
            RuntimeConstantIdxArrayTy, RuntimeConstantsIndicesAlloca,
            {Builder.getInt32(0), Builder.getInt32(ArgI)});
        int ArgNo = JFI.ConstantArgs[ArgI];
        Value *Idx = ConstantInt::get(Builder.getInt32Ty(), ArgNo);
        Builder.CreateStore(Idx, GEP);
      }
      Value *NumRCsValue =
          ConstantInt::get(Builder.getInt32Ty(), NumRuntimeConstants);
      // The prototype is
      // __jit_register_function(const void *HostAddr,
      //                         char const *FunctionName,
      //                         int32_t* RCIndices,
      //                         int32_t NumRCs)
      FunctionType *JitRegisterFunctionFnTy = FunctionType::get(
          VoidType, {VoidPtrType, VoidPtrType, Int32PtrType, Int32Type},
          /* isVarArg=*/false);
      FunctionCallee JitRegisterKernelFn = M.getOrInsertFunction(
          "__jit_register_function", JitRegisterFunctionFnTy);

      constexpr int StubOperand = 1;
      Builder.CreateCall(JitRegisterKernelFn,
                         {RegisterCB->getArgOperand(1),
                          StubToKernelMap[FunctionToRegister],
                          RuntimeConstantsIndicesAlloca, NumRCsValue});
    }
  }

  bool run(Module &M) {
    parseAnnotations(M);

    if (JitFunctionInfoMap.empty())
      return false;

    DEBUG(dbgs() << "=== Pre Original Host Module\n"
                 << M << "=== End of Pre Original Host Module\n");

    // ==================
    // Device compilation
    // ==================

    // For device compilation, just extract the module IR for annotated device
    // kernels and return.
    if (isDeviceCompilation(M)) {
      for (auto &JFI : JitFunctionInfoMap)
        emitJitModuleDevice(M, JFI);

      return true;
    }

    // ================
    // Host compilation
    // ================

    if (hasDeviceLaunchKernelCalls(M)) {
      getKernelHostStubs(M);
      instrumentRegisterVar(M);
      emitModuleUniqueIdGlobal(M);
      instrumentRegisterJITFunc(M);
    }

    for (auto &JFI : JitFunctionInfoMap) {
      Function *JITFn = JFI.first;
      DEBUG(dbgs() << "Processing JIT Function " << JITFn->getName() << "\n");
      if (isDeviceKernelHostStub(M, *JITFn)) {
        emitJitLaunchKernelCall(M, JFI);
        DEBUG(dbgs() << "DONE!\n");

        continue;
      }

      emitJitModuleHost(M, JFI);
      emitJitEntryCall(M, JFI);
    }

    DEBUG(dbgs() << "=== Post Original Host Module\n"
                 << M << "=== End Post Original Host Module\n");
    if (verifyModule(M, &errs()))
      FATAL_ERROR("Broken original module found, compilation aborted!");

    return true;
  }
};

// New PM implementation.
struct ProteusJitPass : PassInfoMixin<ProteusJitPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {
    ProteusJitPassImpl PJP;
    bool Changed = PJP.run(M);
    if (Changed)
      // TODO: is anything preserved?
      return PreservedAnalyses::none();

    return PreservedAnalyses::all();
  }

  // Without isRequired returning true, this pass will be skipped for functions
  // decorated with the optnone LLVM attribute. Note that clang -O0 decorates
  // all functions with optnone.
  static bool isRequired() { return true; }
};

// Legacy PM implementation.
struct LegacyProteusJitPass : public ModulePass {
  static char ID;
  LegacyProteusJitPass() : ModulePass(ID) {}
  bool runOnModule(Module &M) override {
    ProteusJitPassImpl PJP;
    bool Changed = PJP.run(M);
    return Changed;
  }
};
} // namespace

//-----------------------------------------------------------------------------
// New PM Registration
//-----------------------------------------------------------------------------
llvm::PassPluginLibraryInfo getProteusJitPassPluginInfo() {
  const auto Callback = [](PassBuilder &PB) {
    // TODO: decide where to insert it in the pipeline. Early avoids
    // inlining jit function (which disables jit'ing) but may require more
    // optimization, hence overhead, at runtime. We choose after early
    // simplifications which should avoid inlining and present a reasonably
    // analyzable IR module.

    // NOTE: For device jitting it should be possible to register the pass late
    // to reduce compilation time and does lose the kernel due to inlining.
    // However, there are linking errors, working assumption is that the hiprtc
    // linker cannot re-link already linked device libraries and aborts.

    // PB.registerPipelineStartEPCallback(
    // PB.registerOptimizerLastEPCallback(
    PB.registerPipelineEarlySimplificationEPCallback(
        [&](ModulePassManager &MPM, auto) {
          MPM.addPass(ProteusJitPass());
          return true;
        });
  };

  return {LLVM_PLUGIN_API_VERSION, "ProteusJitPass", LLVM_VERSION_STRING,
          Callback};
}

// TODO: use by proteus-jit-pass name.
// This is the core interface for pass plugins. It guarantees that 'opt' will
// be able to recognize ProteusJitPass when added to the pass pipeline on the
// command line, i.e. via '-passes=proteus-jit-pass'
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getProteusJitPassPluginInfo();
}

//-----------------------------------------------------------------------------
// Legacy PM Registration
//-----------------------------------------------------------------------------
// The address of this variable is used to uniquely identify the pass. The
// actual value doesn't matter.
char LegacyProteusJitPass::ID = 0;

// This is the core interface for pass plugins. It guarantees that 'opt' will
// recognize LegacyProteusJitPass when added to the pass pipeline on the command
// line, i.e.  via '--legacy-jit-pass'
static RegisterPass<LegacyProteusJitPass>
    X("legacy-jit-pass", "Jit Pass",
      false, // This pass doesn't modify the CFG => false
      false  // This pass is not a pure analysis pass => false
    );
