//===-- ProteusPass.cpp -- Extact code/runtime info for Proteus JIT --===//
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
//      opt -enable-new-pm=0 -load libProteusPass.dylib -legacy-proteus-pass
//      -disable-output `\`
//        <input-llvm-file>
//    2. New PM
//      opt -load-pass-plugin=libProteusPass.dylib -passes="proteus-pass" `\`
//        -disable-output <input-llvm-file>
//
//
//===----------------------------------------------------------------------===//

#include <string>

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/Analysis/CallGraph.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/CallingConv.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Mangler.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Object/ELF.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/FileSystem/UniqueID.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/MemoryBufferRef.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/AlwaysInliner.h>
#include <llvm/Transforms/IPO/GlobalDCE.h>
#include <llvm/Transforms/IPO/GlobalOpt.h>
#include <llvm/Transforms/IPO/MergeFunctions.h>
#include <llvm/Transforms/IPO/StripDeadPrototypes.h>
#include <llvm/Transforms/IPO/StripSymbols.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>

#include "proteus/Cloning.h"
#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/Error.h"
#include "proteus/Hashing.hpp"
#include "proteus/Logger.hpp"
#include "proteus/RuntimeConstantTypeHelpers.h"

#include "AnnotationHandler.h"
#include "Helpers.h"

using namespace llvm;
using namespace proteus;

//-----------------------------------------------------------------------------
// ProteusPass implementation
//-----------------------------------------------------------------------------
namespace {
static cl::opt<bool> ForceProteusAnnotateAll(
    "force-proteus-jit-annotate-all",
    cl::desc("Apply the 'jit' annotation on all GPU kernels"), cl::init(false));

class ProteusPassImpl {
public:
  ProteusPassImpl(Module &M) : Types(M) {}

  bool run(Module &M, bool IsLTO) {
    AnnotationHandler AnnotHandler{M};
    // We need collect any kernel host stubs to pass to parse annotations, used
    // in forced annotations.
    const auto StubToKernelMap = getKernelHostStubs(M);

    // We force annotate all kernels if the force annotations flag is set and
    // this is not HIP LTO compilation, since LTO constituent modules have
    // already been processed.
    AnnotHandler.parseAnnotations(JitFunctionInfoMap, StubToKernelMap,
                                  (!IsLTO && ForceProteusAnnotateAll));

    DEBUG(Logger::logs("proteus-pass")
          << "=== Pre Original Host Module\n"
          << M << "=== End of Pre Original Host Module\n");

    // ==================
    // Device compilation
    // ==================

    // For device compilation, just extract the module IR of device code
    // and return.
    if (isDeviceCompilation(M)) {
      emitJitModuleDevice(M, IsLTO);

      return true;
    }

    // ================
    // Host compilation
    // ================

    instrumentRegisterLinkedBinary(M);
    instrumentRegisterFatBinary(M);
    instrumentRegisterFatBinaryEnd(M);
    instrumentRegisterVar(M);
    findJitVariables(M);
    registerLambdaFunctions(M);

    if (hasDeviceLaunchKernelCalls(M)) {
      AnnotHandler.parseManifestFileAnnotations(StubToKernelMap,
                                                JitFunctionInfoMap);
      instrumentRegisterFunction(M);
      emitJitLaunchKernelCall(M);
    }

    for (auto &JFI : JitFunctionInfoMap) {
      Function *JITFn = JFI.first;
      DEBUG(Logger::logs("proteus-pass")
            << "Processing JIT Function " << JITFn->getName() << "\n");
      // Skip host device stubs coming from kernel annotations.
      if (isDeviceKernelHostStub(StubToKernelMap, *JITFn))
        continue;

      emitJitModuleHost(M, JFI);
      emitJitEntryCall(M, JFI);
    }

    DEBUG(Logger::logs("proteus-pass")
          << "=== Post Original Host Module\n"
          << M << "=== End Post Original Host Module\n");

    if (verifyModule(M, &errs()))
      PROTEUS_FATAL_ERROR("Broken original module found, compilation aborted!");

    return true;
  }

private:
  ProteusTypes Types;

  MapVector<Function *, JitFunctionInfo> JitFunctionInfoMap;
  SmallPtrSet<Function *, 16> ModuleDeviceKernels;

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
    Passes.addPass(GlobalOptPass());
    Passes.addPass(GlobalDCEPass());
    Passes.addPass(StripDeadDebugInfoPass());
    Passes.addPass(StripDeadPrototypesPass());

    Passes.run(M, MAM);

    StripDebugInfo(M);
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

    DEBUG(Logger::logs("proteus-pass")
          << "=== Pre Passes Host JIT Module\n"
          << *JitMod << "=== End of Pre Passes Host JIT Module\n");

    // Run a global DCE pass and O3 on the JIT module IR to remove unnecessary
    // symbols and reduce the IR to JIT at runtime.
    runCleanupPassPipeline(*JitMod);
    runOptimizationPassPipeline(*JitMod);

    // TODO: Do we want to keep debug info to use for GDB/LLDB
    // interfaces for debugging jitted code?
    StripDebugInfo(*JitMod);

    // Add metadata for the JIT function to store the argument positions for
    // runtime constants.
    emitJitFunctionArgMetadata(*JitMod, JFI, *JitF);

    if (verifyModule(*JitMod, &errs()))
      PROTEUS_FATAL_ERROR("Broken JIT module found, compilation aborted!");

    raw_string_ostream OS(JFI.ModuleIR);
    WriteBitcodeToFile(*JitMod, OS);
    OS.flush();

    DEBUG(Logger::logs("proteus-pass")
          << "=== Final Host JIT Module\n"
          << *JitMod << "=== End of Final Host JIT Module\n");
  }

  // Returns true for globals that are not needed when cloning to extract a
  // module.
  static bool skipGlobal(const GlobalValue &G) {
    if (G.getName().starts_with("_jit_bitcode") ||
        G.getName().starts_with("__clang_gpu_used_external") ||
        G.getName().starts_with("__hip_cuid") ||
        G.getName().starts_with("llvm.used") ||
        G.getName().starts_with("llvm.compiler.used") ||
        G.getName().starts_with("llvm.global.annotations"))
      return true;

    return false;
  }

  // Emit a uniquely named global variable in a corresponding section that
  // contains the embedded bitcode module.
  void emitModuleDevice(Module &M, Module &EmbedM, StringRef Id,
                        bool HasSourceFileID) {
    SmallVector<char> Bitcode;
    raw_svector_ostream OS(Bitcode);
    WriteBitcodeToFile(EmbedM, OS);

    HashT HashValue = hash(StringRef{Bitcode.data(), Bitcode.size()});

    std::string GVName = "_jit_bitcode_" + Id.str() +
                         (HasSourceFileID ? ("_" + getUniqueFileID(M)) : "");
    //  NOTE: HIP compilation supports custom section in the binary to store
    //  the IR. CUDA does not, hence we parse the IR by reading the global
    //  from the device memory.
    Constant *JitModule = ConstantDataArray::get(
        M.getContext(),
        ArrayRef<uint8_t>((const uint8_t *)Bitcode.data(), Bitcode.size()));
    if (M.getNamedGlobal(GVName))
      PROTEUS_FATAL_ERROR(
          "Expected unique name for jit module global variable " + GVName);
    auto *GV =
        new GlobalVariable(M, JitModule->getType(), /* isConstant */ true,
                           GlobalValue::ExternalLinkage, JitModule, GVName);
    appendToUsed(M, {GV});
    // We append the hash value to the section name and retrieve it in HIP JIT
    // compilation to avoid hashing at runtime.
    GV->setSection(".jit.bitcode." + Id.str() + getUniqueModuleId(&M) + "." +
                   HashValue.toString());
    DEBUG(Logger::logs("proteus-pass")
          << "Emit jit bitcode GV " << GVName << "\n");
  }

  void emitLinkedKernelModules(Module &LTOModule) {
    LLVMContext Ctx;
    auto LinkedModule = std::make_unique<Module>("linked.module", Ctx);
    LinkedModule->setSourceFileName("linked.module");
    LinkedModule->setDataLayout(LTOModule.getDataLayout());
    LinkedModule->setTargetTriple(LTOModule.getTargetTriple());
    LinkedModule->setModuleInlineAsm(LTOModule.getModuleInlineAsm());
#if LLVM_VERSION_MAJOR >= 18
    LinkedModule->IsNewDbgInfoFormat = LTOModule.IsNewDbgInfoFormat;
#endif

    // Gather all extracted modules and the pruned LTO module.
    SmallVector<std::unique_ptr<Module>> LinkedModules;
    StringSet KernelSymbols;

    auto ExtractKernelSymbols = [&LTOModule, &KernelSymbols]() {
      for (auto &F : LTOModule) {
        // Skip functions that are not marked for proteus jit.
        if (!F.getMetadata("proteus.jit"))
          continue;

        // Skip non-kernel device functions, for example, device lambda
        // functions.
        if (!isDeviceKernel(&F))
          continue;

        KernelSymbols.insert(F.getName());
      }
    };

    ExtractKernelSymbols();

    SmallPtrSet<GlobalVariable *, 32> RemoveModuleGVs;
    auto ExtractLinkedModules = [&Ctx, &LTOModule, &LinkedModules,
                                 &RemoveModuleGVs]() {
      StringSet DefSet;
      for (auto &GVar : LTOModule.globals()) {
        if (!GVar.hasName())
          continue;

        if (!GVar.getName().starts_with("_jit_bitcode"))
          continue;

        // Found an extracted bitcode global variable, parse the IR and
        // store the names of definitions to prune the LTO module.
        ConstantDataArray *JitModule =
            dyn_cast<ConstantDataArray>(GVar.getInitializer());
        assert(JitModule && "Expected non-null bitcode for JIT module");
        StringRef BitcodeData = JitModule->getAsString();

        // It is important to preserve the unique global variable name as
        // the module name for the parsed module. It will be used later to
        // create unique symbols for linking modules together.
        MemoryBufferRef MBRef{BitcodeData, GVar.getName()};
        auto ExpectedParsedModule = parseBitcodeFile(MBRef, Ctx);
        if (auto E = ExpectedParsedModule.takeError())
          PROTEUS_FATAL_ERROR("Error: " + toString(std::move(E)));
        auto ParsedModule = std::move(*ExpectedParsedModule);
        for (auto &G : ParsedModule->global_values()) {
          if (!G.isDeclaration())
            DefSet.insert(G.getName());
        }

        LinkedModules.push_back(std::move(ParsedModule));

        RemoveModuleGVs.insert(&GVar);
      }

      auto ShouldClone = [&DefSet](const GlobalValue *GV) {
        if (GV->hasName()) {
          if (DefSet.contains(GV->getName()))
            return false;

          if (skipGlobal(*GV))
            return false;
        }

        return true;
      };
      // Create pruned LTO module avoiding unneeded globals and globals in
      // the DefSet. Roundtrip it through bitcode parsing to the top-level
      // context.
      ValueToValueMapTy VMap;
      auto PrunedLTOModule = CloneModule(LTOModule, VMap, ShouldClone);
      for (auto &G : PrunedLTOModule->global_values())
        if (G.hasInternalLinkage())
          G.setLinkage(GlobalValue::ExternalLinkage);

      StripDebugInfo(*PrunedLTOModule);

      if (verifyModule(*PrunedLTOModule, &errs()))
        PROTEUS_FATAL_ERROR(
            "Broken pruned lto module found, compilation aborted!");

      SmallVector<char> Bitcode;
      raw_svector_ostream BOS{Bitcode};
      WriteBitcodeToFile(*PrunedLTOModule, BOS);
      MemoryBufferRef MBRef{StringRef{Bitcode.data(), Bitcode.size()},
                            "pruned.lto.module"};
      auto ExpectedPrunedLTOModuleInCtx = parseBitcodeFile(MBRef, Ctx);
      if (auto E = ExpectedPrunedLTOModuleInCtx.takeError())
        PROTEUS_FATAL_ERROR("Error parsing pruned lto module " +
                            toString(std::move(E)));
      LinkedModules.push_back(std::move(*ExpectedPrunedLTOModuleInCtx));
    };

    ExtractLinkedModules();
    // Remove global vars with per-TU modules as they are not needed anymore.
    // The final step in this method will create per-kernel modules.
    for (auto *GV : RemoveModuleGVs) {
      removeFromUsedLists(LTOModule, [GV](Constant *C) {
        if (GV == C)
          return true;

        return false;
      });
      LTOModule.eraseGlobalVariable(GV);
    }

    Linker IRLinker(*LinkedModule);
    for (auto &Mod : LinkedModules) {
      if (IRLinker.linkInModule(std::move(Mod)))
        PROTEUS_FATAL_ERROR("Linking failed");
    }

    runCleanupPassPipeline(*LinkedModule);

    const char *EnvValue = std::getenv("PROTEUS_PASS_CREATE_KERNEL_MODULES");
    bool CreateKernelModules =
        (EnvValue ? static_cast<bool>(std::stoi(EnvValue)) : true);

    if (CreateKernelModules) {
      for (auto &Sym : KernelSymbols) {
        auto KernelName = Sym.getKey();

        if (!LinkedModule->getFunction(KernelName))
          PROTEUS_FATAL_ERROR("Expected kernel function in linked module");

        auto KernelModule = cloneKernelFromModules({*LinkedModule}, KernelName);
        runCleanupPassPipeline(*KernelModule);

        if (verifyModule(*KernelModule, &errs()))
          PROTEUS_FATAL_ERROR(
              "Broken original module found, compilation aborted!");

        emitModuleDevice(LTOModule, *KernelModule, KernelName, false);
      }
    } else {
      emitModuleDevice(LTOModule, *LinkedModule, "lto", false);
    }
  }

  void emitJitModuleDevice(Module &M, bool IsLTO) {
    SmallVector<char> Bitcode;
    raw_svector_ostream OS(Bitcode);

    // For LTO, supported only in HIP RDC, create the per-kernel linked module
    // AOT.
    if (IsLTO) {
      emitLinkedKernelModules(M);
    } else {
      // Emit the TU-wide extracted module.
      ValueToValueMapTy VMap;
      auto ShouldClone = [](const GlobalValue *G) {
        if (skipGlobal(*G))
          return false;

        return true;
      };
      auto EmitM = CloneModule(M, VMap, ShouldClone);
      runCleanupPassPipeline(*EmitM);

      emitModuleDevice(M, *EmitM, "tu", true);
    }
  }

  void emitJitFunctionArgMetadata(Module &JitMod, JitFunctionInfo &JFI,
                                  Function &JitF) {
    LLVMContext &Ctx = JitMod.getContext();
    SmallVector<Metadata *> ConstArgNos;
    for (auto &RCI : JFI.ConstantArgs) {
      Metadata *Meta = ConstantAsMetadata::get(
          ConstantInt::get(Types.Int32Ty, RCI.ArgInfo.Pos));
      ConstArgNos.push_back(Meta);
    }
    MDNode *Node = MDNode::get(Ctx, ConstArgNos);
    JitF.setMetadata("jit_arg_nos", Node);
  }

  FunctionCallee getJitEntryFn(Module &M) {
    // The prototype is
    // __jit_entry(char *FnName,
    //             char *IR,
    //             int IRSize,
    //             void **Args,
    //             RuntimeConstantInfo **RCInfoArrayPtr,
    //             int32_t NumRCs)

    FunctionType *JitEntryFnTy =
        FunctionType::get(Types.PtrTy,
                          {Types.PtrTy, Types.PtrTy, Types.Int32Ty, Types.PtrTy,
                           Types.PtrTy, Types.Int32Ty},
                          /* isVarArg=*/false);
    FunctionCallee JitEntryFn =
        M.getOrInsertFunction("__jit_entry", JitEntryFnTy);

    return JitEntryFn;
  }

  FunctionCallee getProteusCreateRuntimeConstantInfoScalarFn(Module &M) {
    // RuntimeConstantInfo *__proteus_create_runtime_constant_info(
    //             RuntimeConstantType Type (int32_t),
    //             int32_t Pos)
    return M.getOrInsertFunction(
        "__proteus_create_runtime_constant_info_scalar", Types.PtrTy,
        Types.Int32Ty, Types.Int32Ty);
  }

  FunctionCallee
  getProteusCreateRuntimeConstantInfoArrayConstSizeFn(Module &M) {
    // RuntimeConstantInfo *__proteus_create_runtime_constant_info(
    //             RuntimeConstantType Type (int32_t),
    //             int32_t Pos,
    //             int32_t NumElts,
    //             RuntimeConstantType EltType (int32_t))
    return M.getOrInsertFunction(
        "__proteus_create_runtime_constant_info_array_const_size", Types.PtrTy,
        Types.Int32Ty, Types.Int32Ty, Types.Int32Ty, Types.Int32Ty);
  }

  FunctionCallee
  getProteusCreateRuntimeConstantInfoArrayRunConstSizeFn(Module &M) {
    // RuntimeConstantInfo *__proteus_create_runtime_constant_info(
    //             RuntimeConstantType Type (int32_t),
    //             int32_t Pos,
    //             RuntimeConstantType EltType (int32_t),
    //             RuntimeConstantType NumEltsType (int32_t),
    //             int32_t NumEltsPos)
    return M.getOrInsertFunction(
        "__proteus_create_runtime_constant_info_array_runconst_size",
        Types.PtrTy, Types.Int32Ty, Types.Int32Ty, Types.Int32Ty, Types.Int32Ty,
        Types.Int32Ty);
  }

  FunctionCallee getProteusCreateRuntimeConstantInfoObjectFn(Module &M) {
    // RuntimeConstantInfo *__proteus_create_runtime_constant_info_object(
    //             RuntimeConstantType Type (int32_t),
    //             int32_t Pos,
    //             int32_t Size,
    //             bool PassByValue)
    return M.getOrInsertFunction(
        "__proteus_create_runtime_constant_info_object", Types.PtrTy,
        Types.Int32Ty, Types.Int32Ty, Types.Int32Ty, Types.Int1Ty);
  }

  void emitRuntimeConstantInfoScalar(
      Module &M, const RuntimeConstantInfo &RCInfo, IRBuilderBase &Builder,
      ArrayType *RuntimeConstantInfoPtrArrayTy,
      GlobalVariable *RuntimeConstantInfoPtrArray, size_t Idx) {
    FunctionCallee CreateFn = getProteusCreateRuntimeConstantInfoScalarFn(M);
    Constant *TypeIdC = ConstantInt::get(Types.Int32Ty, RCInfo.ArgInfo.Type);
    Constant *ArgNoC = ConstantInt::get(Types.Int32Ty, RCInfo.ArgInfo.Pos);
    Value *RCInfoPtr = Builder.CreateCall(CreateFn, {TypeIdC, ArgNoC});
    Value *GEP = Builder.CreateGEP(RuntimeConstantInfoPtrArrayTy,
                                   RuntimeConstantInfoPtrArray,
                                   {ConstantInt::get(Types.Int32Ty, 0),
                                    ConstantInt::get(Types.Int32Ty, Idx)});
    Builder.CreateStore(RCInfoPtr, GEP);
  }

  void emitRuntimeConstantInfoArrayConstSize(
      Module &M, const RuntimeConstantInfo &RCInfo, IRBuilderBase &Builder,
      ArrayType *RuntimeConstantInfoPtrArrayTy,
      GlobalVariable *RuntimeConstantInfoPtrArray, size_t Idx) {
    if (!RCInfo.OptArrInfo)
      PROTEUS_FATAL_ERROR("Expected existing array info");

    FunctionCallee CreateFn =
        getProteusCreateRuntimeConstantInfoArrayConstSizeFn(M);
    Constant *TypeIdC = ConstantInt::get(Types.Int32Ty, RCInfo.ArgInfo.Type);
    Constant *ArgNoC = ConstantInt::get(Types.Int32Ty, RCInfo.ArgInfo.Pos);
    Constant *NumEltsC =
        ConstantInt::get(Types.Int32Ty, RCInfo.OptArrInfo->NumElts);
    Constant *EltTypeC =
        ConstantInt::get(Types.Int32Ty, RCInfo.OptArrInfo->EltType);
    Value *RCInfoPtr = Builder.CreateCall(CreateFn, {
                                                        TypeIdC,
                                                        ArgNoC,
                                                        NumEltsC,
                                                        EltTypeC,
                                                    });
    Value *GEP = Builder.CreateGEP(RuntimeConstantInfoPtrArrayTy,
                                   RuntimeConstantInfoPtrArray,
                                   {ConstantInt::get(Types.Int32Ty, 0),
                                    ConstantInt::get(Types.Int32Ty, Idx)});
    Builder.CreateStore(RCInfoPtr, GEP);
  }

  void emitRuntimeConstantInfoArrayRunConstSize(
      Module &M, const RuntimeConstantInfo &RCInfo, IRBuilderBase &Builder,
      ArrayType *RuntimeConstantInfoPtrArrayTy,
      GlobalVariable *RuntimeConstantInfoPtrArray, size_t Idx) {
    if (!RCInfo.OptArrInfo)
      PROTEUS_FATAL_ERROR("Expected array info");

    FunctionCallee CreateFn =
        getProteusCreateRuntimeConstantInfoArrayRunConstSizeFn(M);
    Constant *TypeIdC = ConstantInt::get(Types.Int32Ty, RCInfo.ArgInfo.Type);
    Constant *ArgNoC = ConstantInt::get(Types.Int32Ty, RCInfo.ArgInfo.Pos);
    Constant *EltTypeC =
        ConstantInt::get(Types.Int32Ty, RCInfo.OptArrInfo->EltType);
    Constant *NumEltsTypeC = ConstantInt::get(
        Types.Int32Ty, RCInfo.OptArrInfo->OptNumEltsRCInfo->Type);
    Constant *NumEltsPosC = ConstantInt::get(
        Types.Int32Ty, RCInfo.OptArrInfo->OptNumEltsRCInfo->Pos);
    Value *RCInfoPtr = Builder.CreateCall(
        CreateFn, {TypeIdC, ArgNoC, EltTypeC, NumEltsTypeC, NumEltsPosC});
    Value *GEP = Builder.CreateGEP(RuntimeConstantInfoPtrArrayTy,
                                   RuntimeConstantInfoPtrArray,
                                   {ConstantInt::get(Types.Int32Ty, 0),
                                    ConstantInt::get(Types.Int32Ty, Idx)});
    Builder.CreateStore(RCInfoPtr, GEP);
  }

  void emitRuntimeConstantInfoArray(Module &M,
                                    const RuntimeConstantInfo &RCInfo,
                                    IRBuilderBase &Builder,
                                    ArrayType *RuntimeConstantInfoPtrArrayTy,
                                    GlobalVariable *RuntimeConstantInfoPtrArray,
                                    size_t Idx) {
    if (!RCInfo.OptArrInfo)
      PROTEUS_FATAL_ERROR("Expected array info");

    if (RCInfo.OptArrInfo->OptNumEltsRCInfo) {
      emitRuntimeConstantInfoArrayRunConstSize(
          M, RCInfo, Builder, RuntimeConstantInfoPtrArrayTy,
          RuntimeConstantInfoPtrArray, Idx);
    } else {
      emitRuntimeConstantInfoArrayConstSize(M, RCInfo, Builder,
                                            RuntimeConstantInfoPtrArrayTy,
                                            RuntimeConstantInfoPtrArray, Idx);
    }
  }

  void emitRuntimeConstantInfoObject(
      Module &M, const RuntimeConstantInfo &RCInfo, IRBuilderBase &Builder,
      ArrayType *RuntimeConstantInfoPtrArrayTy,
      GlobalVariable *RuntimeConstantInfoPtrArray, size_t Idx) {
    if (!RCInfo.OptObjInfo)
      PROTEUS_FATAL_ERROR("Expected object info");

    FunctionCallee CreateFn = getProteusCreateRuntimeConstantInfoObjectFn(M);

    Constant *TypeIdC = ConstantInt::get(Types.Int32Ty, RCInfo.ArgInfo.Type);
    Constant *ArgNoC = ConstantInt::get(Types.Int32Ty, RCInfo.ArgInfo.Pos);
    Constant *SizeC = ConstantInt::get(Types.Int32Ty, RCInfo.OptObjInfo->Size);
    Constant *PassByValueC =
        ConstantInt::get(Types.Int1Ty, RCInfo.OptObjInfo->PassByValue);

    Value *RCInfoPtr =
        Builder.CreateCall(CreateFn, {TypeIdC, ArgNoC, SizeC, PassByValueC});
    Value *GEP = Builder.CreateGEP(RuntimeConstantInfoPtrArrayTy,
                                   RuntimeConstantInfoPtrArray,
                                   {ConstantInt::get(Types.Int32Ty, 0),
                                    ConstantInt::get(Types.Int32Ty, Idx)});
    Builder.CreateStore(RCInfoPtr, GEP);
  }

  void emitJitEntryCall(Module &M,
                        std::pair<Function *, JitFunctionInfo> &JITInfo) {

    Function *JITFn = JITInfo.first;
    JitFunctionInfo &JFI = JITInfo.second;
    size_t NumRuntimeConstants = JFI.ConstantArgs.size();

    FunctionCallee JitEntryFn = getJitEntryFn(M);

    // Replaces jit'ed functions in the original module with stubs to call the
    // runtime entry point that compiles and links.
    // Replace jit'ed function with a stub function.
    std::string FnName = JITFn->getName().str();
    JITFn->setName("");
    Function *StubFn = Function::Create(JITFn->getFunctionType(),
                                        JITFn->getLinkage(), FnName, M);
    // We need copy attributes as the can affect affect the ABI, such as
    // 'byval', 'byref' for parameters.
    StubFn->setAttributes(JITFn->getAttributes());
    JITFn->replaceAllUsesWith(StubFn);
    JITFn->eraseFromParent();

    // Replace the body of the jit'ed function to call the jit entry, grab the
    // address of the specialized jit version and execute it.
    auto &Ctx = M.getContext();
    IRBuilder<> Builder(BasicBlock::Create(Ctx, "entry", StubFn));
    BasicBlock *InitBlock =
        BasicBlock::Create(Ctx, "init_rtconst_info", StubFn);
    BasicBlock *ContinueBlock = BasicBlock::Create(Ctx, "continue", StubFn);

    // Create a static flag to track if we've initialized the constants
    GlobalVariable *RCInfoInitialized = new GlobalVariable(
        M, Types.Int1Ty, false, GlobalValue::InternalLinkage,
        ConstantInt::getFalse(Ctx), ".proteus.rtconst.info.inited." + FnName);

    // Create a static global array to store the runtime constant pointers.
    ArrayType *RuntimeConstantInfoPtrArrayTy =
        ArrayType::get(Types.PtrTy, JFI.ConstantArgs.size());
    GlobalVariable *RuntimeConstantInfoPtrArray = new GlobalVariable(
        M, RuntimeConstantInfoPtrArrayTy, false, GlobalValue::InternalLinkage,
        ConstantAggregateZero::get(RuntimeConstantInfoPtrArrayTy),
        ".proteus.rtconst.array." + FnName);

    Builder.CreateCondBr(Builder.CreateLoad(Types.Int1Ty, RCInfoInitialized),
                         ContinueBlock, InitBlock);

    // Initialize rt constants block.
    Builder.SetInsertPoint(InitBlock);

    for (size_t I = 0; I < NumRuntimeConstants; ++I) {
      auto &RCInfo = JFI.ConstantArgs[I];

      if (RCInfo.ArgInfo.Type == RuntimeConstantType::ARRAY) {
        emitRuntimeConstantInfoArray(M, RCInfo, Builder,
                                     RuntimeConstantInfoPtrArrayTy,
                                     RuntimeConstantInfoPtrArray, I);
      } else if ((RCInfo.ArgInfo.Type == RuntimeConstantType::STATIC_ARRAY) ||
                 (RCInfo.ArgInfo.Type == RuntimeConstantType::VECTOR)) {
        emitRuntimeConstantInfoArrayConstSize(M, RCInfo, Builder,
                                              RuntimeConstantInfoPtrArrayTy,
                                              RuntimeConstantInfoPtrArray, I);
      } else if (RCInfo.ArgInfo.Type == RuntimeConstantType::OBJECT) {
        emitRuntimeConstantInfoObject(M, RCInfo, Builder,
                                      RuntimeConstantInfoPtrArrayTy,
                                      RuntimeConstantInfoPtrArray, I);
      } else if (isScalarRuntimeConstantType(RCInfo.ArgInfo.Type)) {
        emitRuntimeConstantInfoScalar(M, RCInfo, Builder,
                                      RuntimeConstantInfoPtrArrayTy,
                                      RuntimeConstantInfoPtrArray, I);
      } else {
        PROTEUS_FATAL_ERROR("Unsupported runtime constant type " +
                            toString(RCInfo.ArgInfo.Type));
      }
    }
    // Mark runtime constants info as initialized.
    Builder.CreateStore(ConstantInt::getTrue(Ctx), RCInfoInitialized);
    // Branch to continue block.
    Builder.CreateBr(ContinueBlock);

    Builder.SetInsertPoint(ContinueBlock);

    // Create globals for the function name, the IR string.
    auto *FnNameGlobal = Builder.CreateGlobalString(StubFn->getName());
    auto *StrIRGlobal = Builder.CreateGlobalString(JFI.ModuleIR);

    ArrayType *ArgPtrsTy = ArrayType::get(Types.PtrTy, StubFn->arg_size());
    Value *ArgPtrs = nullptr;
    if (NumRuntimeConstants > 0) {
      ArgPtrs = Builder.CreateAlloca(ArgPtrsTy);
      // Create an alloca for each argument to store a pointer to the argument,
      // mimicking how arguments are passed for GPU kernels. This is done so
      // that we have a uniform way to read function/kernel arguments in the
      // Proteus runtime for both host and device code.
      SmallVector<AllocaInst *> ArgPtrAllocas;
      for (size_t ArgI = 0; ArgI < StubFn->arg_size(); ++ArgI) {
        auto *Alloca = Builder.CreateAlloca(StubFn->getArg(ArgI)->getType());
        ArgPtrAllocas.push_back(Alloca);
      }
      // Store each a pointer to the argument value to each alloca.
      for (size_t ArgI = 0; ArgI < StubFn->arg_size(); ++ArgI) {
        auto *GEP = Builder.CreateInBoundsGEP(
            ArgPtrsTy, ArgPtrs, {Builder.getInt32(0), Builder.getInt32(ArgI)});
        // If the argument has the 'byval' or 'byref' attribute, we store the
        // pointer to the argument value directly in the alloca, otherwise we
        // store the pointer-to-pointer. This is done to conform to the ABI and
        // correctly reconstruct the runtime value.
        if (StubFn->getArg(ArgI)->hasByValAttr() ||
            StubFn->getArg(ArgI)->hasByRefAttr()) {
          Builder.CreateStore(StubFn->getArg(ArgI), GEP);
        } else {
          Builder.CreateStore(StubFn->getArg(ArgI), ArgPtrAllocas[ArgI]);
          Builder.CreateStore(ArgPtrAllocas[ArgI], GEP);
        }
      }
    } else
      ArgPtrs = Constant::getNullValue(ArgPtrsTy->getPointerTo());

    auto *JitFnPtr =
        Builder.CreateCall(JitEntryFn, {FnNameGlobal, StrIRGlobal,
                                        Builder.getInt32(JFI.ModuleIR.size()),
                                        ArgPtrs, RuntimeConstantInfoPtrArray,
                                        Builder.getInt32(NumRuntimeConstants)});
    SmallVector<Value *, 8> Args;
    for (auto &Arg : StubFn->args())
      Args.push_back(&Arg);
    auto *CI = Builder.CreateCall(StubFn->getFunctionType(), JitFnPtr, Args);

    // We set param attributes for the call to the function pointer returned by
    // the Proteus JIT runtime since they affect the ABI and codegen.
    for (size_t I = 0; I < StubFn->arg_size(); ++I) {
      auto ParamAttrs = StubFn->getAttributes().getParamAttrs(I);
      for (auto A : ParamAttrs)
        CI->addParamAttr(I, A);
    }

    if (StubFn->getReturnType()->isVoidTy())
      Builder.CreateRetVoid();
    else
      Builder.CreateRet(CI);
  }

  Value *getStubGV(Value *Operand) {
    // NOTE: when called by isDeviceKernelHostStub, Operand may not be a global
    // variable point to the stub, so we check and return null in that case.
    Value *V = nullptr;
    if constexpr (PROTEUS_ENABLE_HIP) {
      // NOTE: Hip creates a global named after the device kernel function that
      // points to the host kernel stub. Because of this, we need to unpeel this
      // indirection to use the host kernel stub for finding the device kernel
      // function name global.
      GlobalVariable *IndirectGV = dyn_cast<GlobalVariable>(Operand);
      V = IndirectGV ? IndirectGV->getInitializer() : nullptr;
    } else if constexpr (PROTEUS_ENABLE_CUDA) {
      GlobalValue *DirectGV = dyn_cast<GlobalValue>(Operand);
      V = DirectGV ? DirectGV : nullptr;
    }

    return V;
  }

  DenseMap<Value *, GlobalVariable *> getKernelHostStubs(Module &M) {
    DenseMap<Value *, GlobalVariable *> StubToKernelMap;
    Function *RegisterFunction = nullptr;

    if (!hasDeviceLaunchKernelCalls(M))
      return StubToKernelMap;

    if (!RegisterFunctionName) {
      PROTEUS_FATAL_ERROR("getKernelHostStubs only callable with `EnableHIP or "
                          "EnableCUDA set.");
      return StubToKernelMap;
    }
    RegisterFunction = M.getFunction(RegisterFunctionName);

    if (!RegisterFunction)
      return StubToKernelMap;

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
        DEBUG(Logger::logs("proteus-pass")
              << "StubToKernelMap Key: " << Key->getName() << " -> " << *GV
              << "\n");
      }
    return StubToKernelMap;
  }

  bool isDeviceKernelHostStub(
      const DenseMap<Value *, GlobalVariable *> &StubToKernelMap,
      Function &Fn) {
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

  FunctionCallee getJitLaunchKernelFn(Module &M) {
    FunctionType *JitLaunchKernelFnTy = nullptr;

    assert(LaunchFunctionName && "Expected valid launch function name");
    Function *LaunchKernelFn = M.getFunction(LaunchFunctionName);
    assert(LaunchKernelFn && "Expected non-null launch kernel function");

    // The ABI of __jit_launch_kernel mirrors the device-specific
    // launchKernel. Note the ABI can be different depending on the host
    // architecture.
    JitLaunchKernelFnTy = LaunchKernelFn->getFunctionType();

    if (!JitLaunchKernelFnTy)
      PROTEUS_FATAL_ERROR(
          "Expected non-null jit entry function type, check "
          "PROTEUS_ENABLE_CUDA|PROTEUS_ENABLE_HIP compilation flags "
          "for ProteusPass");

    FunctionCallee JitLaunchKernelFn =
        M.getOrInsertFunction("__jit_launch_kernel", JitLaunchKernelFnTy);

    return JitLaunchKernelFn;
  }

  void replaceWithJitLaunchKernel(Module &M, CallBase *LaunchKernelCB) {
    FunctionCallee JitLaunchKernelFn = getJitLaunchKernelFn(M);

    // Insert before the launch kernel call instruction.
    IRBuilder<> Builder(LaunchKernelCB);
    CallBase *CallOrInvoke = nullptr;

    SmallVector<Value *> Args = {LaunchKernelCB->arg_begin(),
                                 LaunchKernelCB->arg_end()};

    if (isa<CallInst>(LaunchKernelCB)) {
      CallOrInvoke = Builder.CreateCall(JitLaunchKernelFn, Args);
    } else if (auto *InvokeI = dyn_cast<InvokeInst>(LaunchKernelCB)) {
      CallOrInvoke =
          Builder.CreateInvoke(JitLaunchKernelFn, InvokeI->getNormalDest(),
                               InvokeI->getUnwindDest(), Args);
    }

    if (!CallOrInvoke)
      PROTEUS_FATAL_ERROR(
          "Expected non-null jit launch kernel call or invoke, check "
          "PROTEUS_ENABLE_CUDA|PROTEUS_ENABLE_HIP compilation flags "
          "for ProteusPass");

    LaunchKernelCB->replaceAllUsesWith(CallOrInvoke);
    LaunchKernelCB->eraseFromParent();
  }

  void emitJitLaunchKernelCall(Module &M) {
    Function *LaunchKernelFn = nullptr;
    if (!LaunchFunctionName) {
      PROTEUS_FATAL_ERROR(
          "Expected non-null LaunchKernelFn, check "
          "PROTEUS_ENABLE_CUDA|PROTEUS_ENABLE_HIP compilation flags "
          "for ProteusPass");
    }
    LaunchKernelFn = M.getFunction(LaunchFunctionName);
    if (!LaunchKernelFn)
      PROTEUS_FATAL_ERROR(
          "Expected non-null LaunchKernelFn, check "
          "PROTEUS_ENABLE_CUDA|PROTEUS_ENABLE_HIP compilation flags "
          "for ProteusPass");

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

  FunctionCallee getJitRegisterFatBinaryFn(Module &M) {
    FunctionType *JitRegisterFatbinaryFnTy =
        FunctionType::get(Types.VoidTy, {Types.PtrTy, Types.PtrTy, Types.PtrTy},
                          /* isVarArg=*/false);
    FunctionCallee JitRegisterFatbinaryFn = M.getOrInsertFunction(
        "__jit_register_fatbinary", JitRegisterFatbinaryFnTy);

    return JitRegisterFatbinaryFn;
  }

  void instrumentRegisterFatBinary(Module &M) {
    Function *F = nullptr;

    if (!RegisterFatBinaryName)
      return;

    F = M.getFunction(RegisterFatBinaryName);
    if (!F)
      return;

    FunctionCallee JitRegisterFatBinaryFn = getJitRegisterFatBinaryFn(M);

    for (auto *User : F->users()) {
      CallBase *CB = dyn_cast<CallBase>(User);
      if (!CB)
        continue;

      IRBuilder<> Builder(CB->getNextNode());
      Value *FatbinWrapper = CB->getArgOperand(0);

      std::string GVName = "_jit_bitcode_tu_" + getUniqueFileID(M);
      DEBUG(Logger::logs("proteus-pass")
                << "Instrument register fatbinary bitcode GV " << GVName
                << "\n";);
      auto *Arg = Builder.CreateGlobalString(GVName);

      Builder.CreateCall(JitRegisterFatBinaryFn, {CB, FatbinWrapper, Arg});
    }
  }

  FunctionCallee getJitRegisterFatBinaryEndFn(Module &M) {
    FunctionType *JitRegisterFatBinaryEndFnTy =
        FunctionType::get(Types.VoidTy, {Types.PtrTy},
                          /* isVarArg=*/false);
    FunctionCallee JitRegisterFatBinaryEndFn = M.getOrInsertFunction(
        "__jit_register_fatbinary_end", JitRegisterFatBinaryEndFnTy);

    return JitRegisterFatBinaryEndFn;
  }

  void instrumentRegisterFatBinaryEnd(Module &M) {
    // This is CUDA specific.
    if constexpr (!PROTEUS_ENABLE_CUDA) {
      return;
    }

    Function *F = M.getFunction("__cudaRegisterFatBinaryEnd");
    if (!F)
      return;

    FunctionCallee JitRegisterFatBinaryEndFn = getJitRegisterFatBinaryEndFn(M);

    for (auto *User : F->users()) {
      CallBase *CB = dyn_cast<CallBase>(User);
      if (!CB)
        continue;

      IRBuilder<> Builder(CB->getNextNode());
      Value *FatbinWrapper = CB->getArgOperand(0);
      Builder.CreateCall(JitRegisterFatBinaryEndFn, {FatbinWrapper});
    }
  }

  FunctionCallee getJitRegisterLinkedBinaryFn(Module &M) {
    FunctionType *JitRegisterLinkedBinaryFnTy =
        FunctionType::get(Types.VoidTy, {Types.PtrTy, Types.PtrTy},
                          /* isVarArg=*/false);
    FunctionCallee JitRegisteLinkedBinaryrFn = M.getOrInsertFunction(
        "__jit_register_linked_binary", JitRegisterLinkedBinaryFnTy);

    return JitRegisteLinkedBinaryrFn;
  }

  void instrumentRegisterLinkedBinary(Module &M) {
    // This is CUDA specific.
    if constexpr (!PROTEUS_ENABLE_CUDA) {
      return;
    }

    // Note: we check for __cuda_fatibn_wrapper to avoid emitting for the
    // link.stub. It's not strictly necessary since this module will not have a
    // device bitcode to pull and we skip at runtime.
    if (!M.getGlobalVariable("__cuda_fatbin_wrapper", /*AllowInternal=*/true)) {
      DEBUG(Logger::logs("proteus-pass")
                << "Skip " << M.getSourceFileName() << "\n";)
      return;
    }

    FunctionCallee JitRegisterLinkedBinaryFn = getJitRegisterLinkedBinaryFn(M);

    for (auto &F : M.getFunctionList()) {
      if (!F.getName().starts_with("__cudaRegisterLinkedBinary"))
        continue;

      for (auto *User : F.users()) {
        CallBase *CB = dyn_cast<CallBase>(User);
        if (!CB)
          continue;

        IRBuilder<> Builder(CB);
        std::string GVName = "_jit_bitcode_tu_" + getUniqueFileID(M);
        DEBUG(Logger::logs("proteus-pass")
              << "Instrument register linked binary to extract bitcode GV "
              << GVName << "\n");
        auto *Arg = Builder.CreateGlobalString(GVName);
        Builder.CreateCall(JitRegisterLinkedBinaryFn,
                           {CB->getArgOperand(1), Arg});
      }
    }
  }

  FunctionCallee getJitRegisterVarFn(Module &M) {
    // The prototype is
    // __jit_register_var(void *Handle, const void *HostAddr, const char
    // *VarName, uint64_t VarSize).
    FunctionType *JitRegisterVarFnTy = FunctionType::get(
        Types.VoidTy, {Types.PtrTy, Types.PtrTy, Types.PtrTy, Types.Int64Ty},
        /* isVarArg=*/false);
    FunctionCallee JitRegisterVarFn =
        M.getOrInsertFunction("__jit_register_var", JitRegisterVarFnTy);

    return JitRegisterVarFn;
  }

  void instrumentRegisterVar(Module &M) {
    Function *RegisterVarFn = nullptr;
    if (!RegisterVarName)
      return;

    RegisterVarFn = M.getFunction(RegisterVarName);
    if (!RegisterVarFn)
      return;

    FunctionCallee JitRegisterVarFn = getJitRegisterVarFn(M);

    for (User *Usr : RegisterVarFn->users())
      if (CallBase *CB = dyn_cast<CallBase>(Usr)) {
        IRBuilder<> Builder(CB->getNextNode());
        Value *Handle = CB->getArgOperand(0);
        Value *Symbol = CB->getArgOperand(1);
        auto *GV = dyn_cast<GlobalVariable>(Symbol);
        Value *SymbolName = CB->getArgOperand(2);
        Value *SymbolSize = CB->getArgOperand(5);
        Builder.CreateCall(JitRegisterVarFn,
                           {Handle, GV, SymbolName, SymbolSize});
      }
  }

  FunctionCallee getJitRegisterFunctionFn(Module &M) {
    // The prototype is
    // __jit_register_function(void *Handle,
    //                         void *Kernel,
    //                         char const *KernelName,
    //                         RuntimeConstantInfo **RCInfoArrayPtr,
    //                         int32_t NumRCs)
    FunctionType *JitRegisterFunctionFnTy = FunctionType::get(
        Types.VoidTy,
        {Types.PtrTy, Types.PtrTy, Types.PtrTy, Types.PtrTy, Types.Int32Ty},
        /* isVarArg=*/false);
    FunctionCallee JitRegisterKernelFn = M.getOrInsertFunction(
        "__jit_register_function", JitRegisterFunctionFnTy);

    return JitRegisterKernelFn;
  }

  /// instrumentRegisterFunction instruments kernel functions following GPU
  /// runtime registration with __jit_register_function.
  void instrumentRegisterFunction(Module &M) {
    if (!RegisterFunctionName) {
      PROTEUS_FATAL_ERROR(
          "instrumentRegisterJITFunc only callable with `EnableHIP or "
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
        DEBUG(Logger::logs("proteus-pass") << "Not instrumenting device kernel "
                                           << *FunctionToRegister << "\n");
        continue;
      }

      DEBUG(Logger::logs("proteus-pass")
            << "Instrumenting JIT function " << *FunctionToRegister << "\n");
      const auto &JFI = JitFunctionInfoMap[FunctionToRegister];
      size_t NumRuntimeConstants = JFI.ConstantArgs.size();

      ArrayType *RuntimeConstantInfoPtrArrayTy =
          ArrayType::get(Types.PtrTy, NumRuntimeConstants);

      GlobalVariable *RuntimeConstantInfoPtrArray = new GlobalVariable(
          M, RuntimeConstantInfoPtrArrayTy, false, GlobalValue::InternalLinkage,
          ConstantAggregateZero::get(RuntimeConstantInfoPtrArrayTy),
          ".proteus.rtconst.array." + FunctionToRegister->getName());

      IRBuilder<> Builder(RegisterCB->getNextNode());
      for (size_t I = 0; I < NumRuntimeConstants; ++I) {
        auto &RCInfo = JFI.ConstantArgs[I];

        if (RCInfo.ArgInfo.Type == RuntimeConstantType::ARRAY) {
          emitRuntimeConstantInfoArray(M, RCInfo, Builder,
                                       RuntimeConstantInfoPtrArrayTy,
                                       RuntimeConstantInfoPtrArray, I);
        } else if (RCInfo.ArgInfo.Type == RuntimeConstantType::OBJECT) {
          emitRuntimeConstantInfoObject(M, RCInfo, Builder,
                                        RuntimeConstantInfoPtrArrayTy,
                                        RuntimeConstantInfoPtrArray, I);
        } else if (isScalarRuntimeConstantType(RCInfo.ArgInfo.Type)) {
          emitRuntimeConstantInfoScalar(M, RCInfo, Builder,
                                        RuntimeConstantInfoPtrArrayTy,
                                        RuntimeConstantInfoPtrArray, I);
        } else {
          PROTEUS_FATAL_ERROR("Unsupported runtime constant type " +
                              toString(RCInfo.ArgInfo.Type));
        }
      }

      Value *NumRCsValue =
          ConstantInt::get(Builder.getInt32Ty(), NumRuntimeConstants);

      FunctionCallee JitRegisterFunction = getJitRegisterFunctionFn(M);

      Builder.CreateCall(JitRegisterFunction,
                         {RegisterCB->getArgOperand(0),
                          RegisterCB->getArgOperand(1),
                          RegisterCB->getArgOperand(2),
                          RuntimeConstantInfoPtrArray, NumRCsValue});
    }
  }

  void findJitVariables(Module &M) {
    DEBUG(Logger::logs("proteus-pass") << "finding jit variables" << "\n");
    DEBUG(Logger::logs("proteus-pass") << "users..." << "\n");

    SmallVector<Function *, 16> JitFunctions;

    for (auto &F : M.getFunctionList()) {
      std::string DemangledName = demangle(F.getName().str());
      if (StringRef{DemangledName}.contains("proteus::jit_variable")) {
        JitFunctions.push_back(&F);
      }
    }

    auto FindStorePtr = [&](CallBase *CB) {
      // Find the store instruction user of the JitVariableCB to extract the
      // pointer to the lambda anonymous class object.
      Value *Ptr = nullptr;
      Value *V = CB;
      while (!Ptr) {
        if (!V->hasOneUser())
          PROTEUS_FATAL_ERROR("Expected single user");

        StoreInst *S = dyn_cast<StoreInst>(*(V->users().begin()));
        if (S) {
          DEBUG(Logger::logs("proteus-pass") << "store: " << *S << "\n");
          Ptr = S->getPointerOperand();
          break;
        }

        // Recurse to the next user.
        V = *V->users().begin();
      }

      return Ptr;
    };

    for (auto *Function : JitFunctions) {
      for (auto *User : Function->users()) {
        CallBase *CB = dyn_cast<CallBase>(User);
        if (!CB)
          PROTEUS_FATAL_ERROR(
              "Expected CallBase as user of proteus::jit_variable function");

        DEBUG(Logger::logs("proteus-pass") << "call: " << *CB << "\n");

        Value *V = FindStorePtr(CB);

        GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(V);
        if (GEP) {
          DEBUG(Logger::logs("proteus-pass") << "gep: " << *GEP << "\n");
          auto *Slot = GEP->getOperand(GEP->getNumOperands() - 1);
          DEBUG(Logger::logs("proteus-pass") << "slot: " << *Slot << "\n");
          CB->setArgOperand(1, Slot);
          auto *GEPTy = GEP->getSourceElementType();
          StructType *STy = dyn_cast<StructType>(GEPTy);
          if (!STy)
            PROTEUS_FATAL_ERROR("Expected struct type for lambda");
          const StructLayout *SL = M.getDataLayout().getStructLayout(STy);
          ConstantInt *SlotC = dyn_cast<ConstantInt>(Slot);
          if (!SlotC)
            PROTEUS_FATAL_ERROR("Expected constant slot");
          auto Offset = SL->getElementOffset(SlotC->getZExtValue());
          Constant *OffsetCI = ConstantInt::get(Types.Int32Ty, Offset);
          CB->setArgOperand(2, OffsetCI);
        } else {
          DEBUG(Logger::logs("proteus-pass")
                << "no gep, assuming slot 0" << "\n");
          Constant *C = ConstantInt::get(Types.Int32Ty, 0);
          CB->setArgOperand(1, C);
          CB->setArgOperand(2, C);
        }
      }
    }
  }

  StringRef parseLambdaType(StringRef DemangledName) {
    int L = -1;
    int R = -1;
    int Level = 0;
    // Start after the function symbol to avoid parsing its templated return
    // type.
    size_t Start = DemangledName.find("proteus::register_lambda");
    for (size_t I = Start, E = DemangledName.size(); I < E; ++I) {
      const char C = DemangledName[I];
      if (C == '<') {
        Level++;
        if (Level == 1)
          L = I;
      }

      if (C == '>') {
        if (Level == 1) {
          R = I;
          break;
        }
        Level--;
      }
    }

    assert(L > 0 && R > L && "Expected non-zero L, R for slicing");
    // Remove reference character '&', if it exists.
    if (DemangledName[R - 1] == '&')
      R--;
    // Slicing returns characters [Start, End).
    return DemangledName.slice(L + 1, R);
    ;
  }

  void registerLambdaFunctions(Module &M) {
    DEBUG(Logger::logs("proteus-pass")
          << "registering lambda functions" << "\n");

    SmallVector<Function *, 16> LambdaFunctions;
    for (auto &F : M.getFunctionList()) {
      if (StringRef{demangle(F.getName().str())}.contains(
              "proteus::register_lambda")) {
        LambdaFunctions.push_back(&F);
      }
    }

    for (auto *Function : LambdaFunctions) {
      auto DemangledName = llvm::demangle(Function->getName().str());
      StringRef LambdaType = parseLambdaType(DemangledName);

      DEBUG(Logger::logs("proteus-pass")
            << Function->getName() << " " << DemangledName << " " << LambdaType
            << "\n");

      for (auto *User : Function->users()) {
        CallBase *CB = dyn_cast<CallBase>(User);
        if (!CB)
          PROTEUS_FATAL_ERROR("Expected CallBase as user of "
                              "proteus::register_lambda function");

        IRBuilder<> Builder(CB);
        auto *LambdaNameGlobal = Builder.CreateGlobalString(LambdaType);
        CB->setArgOperand(1, LambdaNameGlobal);
      }
    }
  }
};

// New PM implementation.
struct ProteusPass : PassInfoMixin<ProteusPass> {
  ProteusPass(bool IsLTO) : IsLTO(IsLTO) {}
  bool IsLTO;

  PreservedAnalyses run(Module &M, ModuleAnalysisManager & /*AM*/) {
    ProteusPassImpl PPI{M};

    bool Changed = PPI.run(M, IsLTO);
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
struct LegacyProteusPass : public ModulePass {
  static char ID;
  LegacyProteusPass() : ModulePass(ID) {}
  bool runOnModule(Module &M) override {
    ProteusPassImpl PPI{M};
    bool Changed = PPI.run(M, false);
    return Changed;
  }
};
} // namespace

//-----------------------------------------------------------------------------
// New PM Registration
//-----------------------------------------------------------------------------
llvm::PassPluginLibraryInfo getProteusPassPluginInfo() {
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
          MPM.addPass(ProteusPass{false});
          return true;
        });

    PB.registerFullLinkTimeOptimizationEarlyEPCallback(
        [&](ModulePassManager &MPM, auto) {
          MPM.addPass(ProteusPass{true});
          return true;
        });
  };

  return {LLVM_PLUGIN_API_VERSION, "ProteusPass", LLVM_VERSION_STRING,
          Callback};
}

// This is the core interface for pass plugins. It guarantees that 'opt' will
// be able to recognize ProteusPass when added to the pass pipeline on the
// command line, i.e. via '-passes=proteus-pass'
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getProteusPassPluginInfo();
}

//-----------------------------------------------------------------------------
// Legacy PM Registration
//-----------------------------------------------------------------------------
// The address of this variable is used to uniquely identify the pass. The
// actual value doesn't matter.
char LegacyProteusPass::ID = 0;

// This is the core interface for pass plugins. It guarantees that 'opt' will
// recognize LegacyProteusPass when added to the pass pipeline on the command
// line, i.e.  via '--legacy-proteus-pass'
static RegisterPass<LegacyProteusPass>
    X("legacy-proteuss-pass", "Proteus Pass",
      false, // This pass doesn't modify the CFG => false
      false  // This pass is not a pure analysis pass => false
    );
