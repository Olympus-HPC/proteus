//===-- LambdaAnalysis.cpp -- Extact code/runtime info for Proteus JIT --===//
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
//      opt -enable-new-pm=0 -load libLambdaPass.dylib -legacy-lambda-pass
//      -disable-output `\`
//        <input-llvm-file>
//    2. New PM
//      opt -load-pass-plugin=libLambdaPass.dylib -passes="lambda-pass" `\`
//        -disable-output <input-llvm-file>
//
//
//===----------------------------------------------------------------------===//

#include "AnnotationHandler.h"
#include "Helpers.h"

#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/Error.h"
#include "proteus/Frontend/IRFunction.h"
#include "proteus/impl/Cloning.h"
#include "proteus/impl/Hashing.h"
#include "proteus/impl/Logger.h"
#include "proteus/impl/RuntimeConstantTypeHelpers.h"

#include <cstddef>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/Analysis/CallGraph.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/CallingConv.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Mangler.h>
#include <llvm/IR/Metadata.h>
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

#include <limits>
#include <queue>
#include <string>

using namespace llvm;
using namespace proteus;

//-----------------------------------------------------------------------------
// LambdaPass implementation
//-----------------------------------------------------------------------------
namespace {

struct LambdaJitVariableInfo {
  llvm::Value *Slot;
  llvm::Value *Offset;
  llvm::Type *Type;
};

std::optional<std::string> getGlobalString(const GlobalVariable *GV) {
  if (!GV || !GV->hasInitializer())
    return std::nullopt;

  const Constant *Init = GV->getInitializer();

  if (auto *CDS = dyn_cast<ConstantDataSequential>(Init))
    if (CDS->isString())
      return CDS->getAsCString().str(); // drops trailing '\0'

  if (auto *CDA = dyn_cast<ConstantDataArray>(Init))
    if (CDA->isString())
      return CDA->getAsCString().str();

  return std::nullopt;
}

class LambdaPassImpl {
public:
  LambdaPassImpl(Module &M) : Types(M) {}

  bool run(Module &M, bool IsLTO) {
    AnnotationHandler AnnotHandler{M};
    // Attach Metadata nodes corresponding to llvm.global.annotations to
    // individual function calls.
    materializeWrapperCallMetadata(M);
    // We need collect any kernel host stubs to pass to parse annotations, used
    // in forced annotations.
    const auto StubToKernelMap = getKernelHostStubs(M);

    DenseMap<StructType *, SmallVector<LambdaJitVariableInfo, 16>>
        LambdaStorageTypeToJitIndices;
    DenseMap<Type *, GlobalVariable *> LambdaTypeToGlobalName;

    DenseMap<StructType *, StructType *> FunctorTypeToLambdaType;
    DenseMap<StructType *, StructType *> LambdaTypeToFunctorType;

    // new pipeline: LambdaAnalysis sees early LLVM IR emitted with
    // proteus__register_lambda --> proteus::__register_lambda_impl (1) make a
    // DenseMap from lambda functor types to their underlying storage class (2)
    // Find jit_variable calls and associate with class.anon types (3) Look at
    // register lambda calls, identify the call operators of the functors from
    // the callsite
    //      (a) collect lambda ops in a map to functor ops, if there are two
    //      functor using the same one we need to clone and inject the
    //          clone at the callsite
    //     (b) make the call operators strings. inject register_jit_variable
    //     with the call operator functor string as the key
    // (4) (RUNTIME) Look at all the call operators within a kernel, replace
    // with runtime constants now propagated from the registry.
    //
    // NOTE: We need this for both host and device modules, since the runtime
    // specialization paths expect the function metadata to be present after
    // cloning/extraction.
    makeLambdaCallsUniquePerFunctorOperator(M);
    if (!isDeviceCompilation(M)) {
      getUnderlyingLambdaTypeFromFunctors(M, FunctorTypeToLambdaType,
                                          LambdaTypeToFunctorType);
      instrumentJitVariableStructIndex(M, LambdaTypeToFunctorType,
                                       LambdaStorageTypeToJitIndices);
      registerLambdaFunctions(M, LambdaStorageTypeToJitIndices);
    }

    DEBUG(Logger::logs("lambda-pass")
          << "=== Post Original Host Module\n"
          << M << "=== End Post Original Host Module\n");

    if (verifyModule(M, &errs()))
      reportFatalError("Broken original module found, compilation aborted!");

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

  Value *getStubGV([[maybe_unused]] Value *Operand) {
    // NOTE: when called by isDeviceKernelHostStub, Operand may not be a global
    // variable point to the stub, so we check and return null in that case.
    Value *V = nullptr;
#if PROTEUS_ENABLE_HIP
    // NOTE: Hip creates a global named after the device kernel function that
    // points to the host kernel stub. Because of this, we need to unpeel this
    // indirection to use the host kernel stub for finding the device kernel
    // function name global.
    GlobalVariable *IndirectGV = dyn_cast<GlobalVariable>(Operand);
    V = IndirectGV ? IndirectGV->getInitializer() : nullptr;
#elif PROTEUS_ENABLE_CUDA
    GlobalValue *DirectGV = dyn_cast<GlobalValue>(Operand);
    V = DirectGV ? DirectGV : nullptr;
#endif

    return V;
  }

  static void findAnnotatedFunctions(
      llvm::Module &M, llvm::StringRef Wanted,
      llvm::SmallVectorImpl<std::pair<llvm::Function *, std::uint64_t>> &Out) {
    auto *GA = M.getGlobalVariable("llvm.global.annotations");
    if (!GA)
      return;

    auto *CA = llvm::dyn_cast<llvm::ConstantArray>(GA->getOperand(0));
    if (!CA)
      return;

    llvm::SmallDenseMap<llvm::Function *, std::uint64_t, 32> Seen;

    for (llvm::Value *Elt : CA->operands()) {
      auto *CS = llvm::dyn_cast<llvm::ConstantStruct>(Elt);
      if (!CS || CS->getNumOperands() < 5)
        continue;

      auto *F = llvm::dyn_cast<llvm::Function>(
          CS->getOperand(0)->stripPointerCasts());
      if (!F)
        continue;

      llvm::StringRef Ann;
      if (!llvm::getConstantStringInfo(CS->getOperand(1)->stripPointerCasts(),
                                       Ann))
        continue;
      if (Ann != Wanted)
        continue;

      // Parse the u64 payload (operand 4)
      std::uint64_t Id = 0;
      llvm::Value *ArgsPtr = CS->getOperand(4)->stripPointerCasts();
      if (llvm::isa<llvm::ConstantPointerNull>(ArgsPtr))
        continue;

      auto *ArgsGV = llvm::dyn_cast<llvm::GlobalVariable>(ArgsPtr);
      if (!ArgsGV)
        continue;

      auto *ArgsInit = ArgsGV->getInitializer();
      auto *ArgsCS = llvm::dyn_cast<llvm::ConstantStruct>(ArgsInit);
      if (!ArgsCS || ArgsCS->getNumOperands() < 1)
        continue;

      auto *CI = llvm::dyn_cast<llvm::ConstantInt>(ArgsCS->getOperand(0));
      if (!CI)
        continue;

      Id = CI->getZExtValue();

      if (Seen.try_emplace(F, Id).second)
        Out.emplace_back(F, Id);
    }
  }

  void setRegisteredLambdaMetadata(Function *F, uint64_t Id) {
    setU64FunctionMetadata(F, "proteus.registered_lambda", Id);
  }

  void setU64FunctionMetadata(Function *F, StringRef Key, uint64_t Id) {
    if (!F)
      return;

    if (auto *Existing = F->getMetadata(Key)) {
      if (Existing->getNumOperands() < 1)
        return;
      auto *CAM = dyn_cast<ConstantAsMetadata>(Existing->getOperand(0));
      auto *CI = CAM ? dyn_cast<ConstantInt>(CAM->getValue()) : nullptr;
      if (!CI)
        return;
      if (CI->getZExtValue() != Id)
        reportFatalError("Conflicting u64 metadata for " + Key.str() +
                         " on function " + F->getName().str());
      return;
    }

    LLVMContext &Ctx = F->getContext();
    auto *IdMD =
        ConstantAsMetadata::get(ConstantInt::get(Type::getInt64Ty(Ctx), Id));
    F->setMetadata(Key, MDNode::get(Ctx, {IdMD}));
  }

  void materializeWrapperCallMetadata(Module &M) {
    SmallVector<std::pair<Function *, uint64_t>, 16> WrapperCallFunctions;
    findAnnotatedFunctions(M, "proteus.wrapper_call", WrapperCallFunctions);
    for (auto &[F, Id] : WrapperCallFunctions)
      setU64FunctionMetadata(F, "proteus.wrapper_call", Id);
  }

  Function *cloneLambdaOperator(Function *LambdaOperator, uint64_t Suffix) {
    ValueToValueMapTy VMap;

    Function *NewFunc = Function::Create(
        LambdaOperator->getFunctionType(), LambdaOperator->getLinkage(),
        LambdaOperator->getName() + "_clone_" + std::to_string(Suffix),
        LambdaOperator->getParent());

    auto NewArgIt = NewFunc->arg_begin();
    for (auto &OldArg : LambdaOperator->args()) {
      NewArgIt->setName(OldArg.getName());
      VMap[&OldArg] = &(*NewArgIt++);
    }

    SmallVector<ReturnInst *, 8> Returns;
    CloneFunctionInto(NewFunc, LambdaOperator, VMap,
                      CloneFunctionChangeType::LocalChangesOnly, Returns);

    return NewFunc;
  }

  void makeLambdaCallsUniquePerFunctorOperator(Module &M) {
    SmallVector<std::pair<Function *, uint64_t>> FunctorOperatorMethods;
    DenseMap<Function *, SmallVector<std::pair<CallBase *, uint64_t>, 8>>
        LambdaCallToFunctor;
    findAnnotatedFunctions(M, "proteus.wrapper_call", FunctorOperatorMethods);

    for (auto &[FunctorOperator, FunctorId] : FunctorOperatorMethods) {
      setU64FunctionMetadata(FunctorOperator, "proteus.wrapper_call",
                             FunctorId);
      for (auto &BB : *FunctorOperator) {
        for (auto &I : BB) {
          CallBase *CB = dyn_cast<CallBase>(&I);
          if (!CB)
            continue;
          Function *Callee = CB->getCalledFunction();
          if (!Callee)
            continue;
          std::string CalledFunctionName = Callee->getName().str();
          std::string DemangledCalledFunctionName =
              demangle(CalledFunctionName);
          bool isCallOp = DemangledCalledFunctionName.find("operator()") !=
                          std::string::npos;

          bool looksLikeLambda =
              DemangledCalledFunctionName.find("{lambda") !=
                  std::string::npos || // common demangle form
              DemangledCalledFunctionName.find("::$_") !=
                  std::string::npos || // your main::$_0 form
              CalledFunctionName.find("$_") !=
                  std::string::npos || // survives even if demangle fails
              llvm::StringRef(CalledFunctionName)
                  .starts_with("_ZZ"); // local-scope Itanium

          if (!isCallOp || !looksLikeLambda)
            continue;

          LambdaCallToFunctor[Callee].emplace_back(CB, FunctorId);
        }
      }
    }

    DenseMap<Function *, uint64_t> FunctionsToAnnotate;
    for (auto &Entry : LambdaCallToFunctor) {
      Function *LambdaOperator = Entry.first;
      auto &LambdaOperatorCBVec = Entry.second;

      DenseMap<uint64_t, SmallVector<CallBase *, 8>> CallsByFunctorId;
      for (auto &[LambdaOpCB, CallerId] : LambdaOperatorCBVec)
        CallsByFunctorId[CallerId].push_back(LambdaOpCB);

      if (CallsByFunctorId.empty())
        continue;

      uint64_t PrimaryFunctorId = std::numeric_limits<uint64_t>::max();
      // Ensure a deterministic ordering of assignment
      for (auto &KV : CallsByFunctorId)
        if (KV.first < PrimaryFunctorId)
          PrimaryFunctorId = KV.first;

      FunctionsToAnnotate.try_emplace(LambdaOperator, PrimaryFunctorId);

      for (auto &KV : CallsByFunctorId) {
        uint64_t CallerId = KV.first;
        auto &Calls = KV.second;
        if (CallerId == PrimaryFunctorId)
          continue;

        Function *NewOperator = cloneLambdaOperator(LambdaOperator, CallerId);
        for (CallBase *LambdaOpCB : Calls)
          LambdaOpCB->setCalledFunction(NewOperator);

        FunctionsToAnnotate.try_emplace(NewOperator, CallerId);
      }
    }

    for (auto &KV : FunctionsToAnnotate) {
      Function *F = KV.first;
      uint64_t Id = KV.second;
      setRegisteredLambdaMetadata(F, Id);
    }
  }

  DenseMap<Value *, GlobalVariable *> getKernelHostStubs(Module &M) {
    DenseMap<Value *, GlobalVariable *> StubToKernelMap;
    Function *RegisterFunction = nullptr;

    if (!hasDeviceLaunchKernelCalls(M))
      return StubToKernelMap;

    if (!RegisterFunctionName) {
      reportFatalError("getKernelHostStubs only callable with `EnableHIP or "
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
        StubToKernelMap[Key->stripPointerCasts()] = GV;
        DEBUG(Logger::logs("lambda-pass")
              << "StubToKernelMap Key: " << Key->getName() << " -> " << *GV
              << "\n");
      }
    return StubToKernelMap;
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

  void getUnderlyingLambdaTypeFromFunctors(
      Module &M, DenseMap<StructType *, StructType *> &FunctorToLambdaType,
      DenseMap<StructType *, StructType *> &LambdaToFunctorType) {
    for (llvm::StructType *ST : M.getIdentifiedStructTypes()) {
      if (!ST->getStructName().contains("LambdaFunctorWrapper"))
        continue;
      auto *UnderlyingStruct = dyn_cast<StructType>(ST->getElementType(0));
      if (!UnderlyingStruct)
        reportFatalError("Internal Proteus Error: LambdaFunctorWrapper "
                         "underlying class not initialized correctly.");
      FunctorToLambdaType[ST] = UnderlyingStruct;
      LambdaToFunctorType[UnderlyingStruct] = ST;
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

  FunctionCallee getJitRegisterLambdaRuntimeConstant(Module &M) {
    // __jit_register_variable_instance_raw(
    //   int32_t Type, int32_t Pos, int32_t Offset,
    //   void const* ValuePtr, int32_t Size,
    //   char const* LambdaType, void const* Key)
    FunctionType *FnTy =
        FunctionType::get(Types.VoidTy,
                          {Types.Int32Ty, Types.Int32Ty, Types.Int32Ty,
                           Types.PtrTy, Types.Int64Ty},
                          /*isVarArg=*/false);
    return M.getOrInsertFunction("__jit_register_lambda_runtime_constant",
                                 FnTy);
  }

  std::optional<RuntimeConstantType> getRCTypeForLLVMType(Type *Ty) {
    if (Ty->isIntegerTy(1))
      return RuntimeConstantType::BOOL;
    if (Ty->isIntegerTy(8))
      return RuntimeConstantType::INT8;
    if (Ty->isIntegerTy(32))
      return RuntimeConstantType::INT32;
    if (Ty->isIntegerTy(64))
      return RuntimeConstantType::INT64;
    if (Ty->isFloatTy())
      return RuntimeConstantType::FLOAT;
    if (Ty->isDoubleTy())
      return RuntimeConstantType::DOUBLE;
    if (Ty->isPointerTy())
      return RuntimeConstantType::PTR;
    return std::nullopt;
  }

  /// instrumentJitVariableStructIndex modifies calls to proteus::jit_variable
  /// by injecting the corresponding jit variable offset into the call.
  /// todo(bowen): add assert on jit_variable having an alloca within the same
  /// procedure
  void instrumentJitVariableStructIndex(
      Module &M, DenseMap<StructType *, StructType *> &LambdaTypeToFunctorType,
      DenseMap<StructType *, SmallVector<LambdaJitVariableInfo, 16>>
          &JitIndices) {
    DEBUG(Logger::logs("lambda-pass") << "finding jit variables" << "\n");
    DEBUG(Logger::logs("lambda-pass") << "users..." << "\n");

    SmallVector<Function *, 16> JitFunctions;

    for (auto &F : M.getFunctionList()) {
      std::string DemangledName = demangle(F.getName().str());
      if (StringRef{DemangledName}.contains("proteus::jit_variable")) {
        JitFunctions.push_back(&F);
      }
    }

    for (auto &F : JitFunctions) {
      FunctionType *FTy = F->getFunctionType();
      auto *LoadType = FTy->getParamType(0);
      if (!LoadType)
        reportFatalError("jit function return type null??\n");
      std::queue<Value *> WorkList;
      DenseSet<Value *> Discovered;
      for (auto *Usr : F->users()) {
        CallBase *CB = dyn_cast<CallBase>(Usr);
        if (!CB)
          continue;
        for (auto *Usr : CB->users())
          WorkList.push(Usr);
      }
      // These two integers need to be equivalent--we track down each callsite
      // to an offset in the struct.
      uint16_t numJitVars = WorkList.size();
      uint16_t numSlotsFound = 0;

      while (!WorkList.empty()) {
        Value *CurVal = WorkList.front();
        if (Discovered.contains(CurVal))
          continue;
        Discovered.insert(CurVal);
        WorkList.pop();
        // very simple use-def traversal only handles three cases, which we
        // expect at startEPCallBack
        if (StoreInst *Store = dyn_cast<StoreInst>(CurVal))
          WorkList.push(Store->getPointerOperand());
        if (AllocaInst *Alloca = dyn_cast<AllocaInst>(CurVal)) {
          Constant *Slot = ConstantInt::get(Types.Int32Ty, 0);
          auto *PossibleLamType =
              dyn_cast<StructType>(Alloca->getAllocatedType());
          if (!PossibleLamType ||
              !LambdaTypeToFunctorType.contains(PossibleLamType))
            continue;
          ++numSlotsFound;
          JitIndices[PossibleLamType].emplace_back(
              LambdaJitVariableInfo{Slot, Slot, LoadType});
          if (!LoadType)
            reportFatalError("Load type is null\n");
        }
        if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(CurVal)) {
          StructType *PossibleLamType =
              dyn_cast<StructType>(GEP->getSourceElementType());
          if (!PossibleLamType ||
              !LambdaTypeToFunctorType.contains(PossibleLamType))
            continue;
          auto *Slot = GEP->getOperand(GEP->getNumOperands() - 1);
          const StructLayout *SL =
              M.getDataLayout().getStructLayout(PossibleLamType);
          ConstantInt *SlotC = dyn_cast<ConstantInt>(Slot);
          if (!SlotC)
            reportFatalError("Expected constant slot");

          auto Offset = SL->getElementOffset(SlotC->getZExtValue());
          Constant *OffsetCI = ConstantInt::get(Types.Int32Ty, Offset);
          // Type* LoadType = GEP->getAccessType();
          JitIndices[PossibleLamType].emplace_back(
              LambdaJitVariableInfo{Slot, OffsetCI, LoadType});
          ++numSlotsFound;
        }
      }

      if (numJitVars != numSlotsFound)
        reportFatalError(
            "Analysis found jit_variable callsite outside lambda capture list");
    }
  }

  void registerLambdaFunctions(
      Module &M, DenseMap<StructType *, SmallVector<LambdaJitVariableInfo, 16>>
                     &LambdaStorageTypeToJitIndices) {
    if (LambdaStorageTypeToJitIndices.empty())
      return;
    DEBUG(Logger::logs("lambda-pass")
          << "registering lambda functions" << "\n");
    SmallVector<std::pair<Function *, uint64_t>, 16> RegisterFunctions;
    findAnnotatedFunctions(M, "proteus.register_call", RegisterFunctions);
    for (auto [Function, ID] : RegisterFunctions) {
      SmallVector<ReturnInst *> Rets;
      AllocaInst *AnonClassAlloca = nullptr;
      for (auto &BB : *Function) {
        if (auto *RI = llvm::dyn_cast<llvm::ReturnInst>(BB.getTerminator()))
          Rets.push_back(RI);
        for (Instruction &I : BB) {
          auto *Alloca = dyn_cast<AllocaInst>(&I);
          if (!Alloca)
            continue;
          auto *AllocaStructType =
              dyn_cast<StructType>(Alloca->getAllocatedType());
          if (!AllocaStructType ||
              !LambdaStorageTypeToJitIndices.contains(AllocaStructType))
            continue;
          AnonClassAlloca = Alloca;
        }
      }
      if (Rets.size() != 1)
        reportFatalError("internal proteus error: expected single return from "
                         "register func");
      if (!AnonClassAlloca)
        reportFatalError("internal proteus error: expected anon class alloca");
      StructType *RegisterFuncAllocatedType =
          dyn_cast<StructType>(AnonClassAlloca->getAllocatedType());
      if (!RegisterFuncAllocatedType)
        reportFatalError("internal proteus error: error in logic of "
                         "register_lambda analysis");
      ReturnInst *RetInst = Rets[0];

      // Insert __jit_push_lambda_runtime_constant calls for each lambda type
      // participating in this kernel launch.
      IRBuilder<> Builder(RetInst);
      auto JitVarFn = getJitRegisterLambdaRuntimeConstant(M);
      for (auto JitVarInfo :
           LambdaStorageTypeToJitIndices.find(RegisterFuncAllocatedType)
               ->getSecond()) {
        if (!JitVarInfo.Type)
          reportFatalError("jit var info type is null\n");
        auto ProteusRCType = getRCTypeForLLVMType(JitVarInfo.Type);
        ConstantInt *SlotC = dyn_cast<ConstantInt>(JitVarInfo.Slot);
        if (!SlotC)
          reportFatalError("Expected constant slot");
        const auto Idx = SlotC->getZExtValue();
        if (!ProteusRCType)
          reportFatalError("Proteus does not support user-specified type as a "
                           "jit::variable");

        Value *FieldPtr = Builder.CreateStructGEP(RegisterFuncAllocatedType,
                                                  AnonClassAlloca, Idx);
        Builder.CreateCall(
            JitVarFn,
            {Builder.getInt32(static_cast<int32_t>(ProteusRCType.value())),
             Builder.getInt32(Idx), JitVarInfo.Offset, FieldPtr,
             Builder.getInt64(ID)});
      }
    }
  }

  // Detect a CUDA module by the presence of the __cuda_register_globals
  // function.
  bool isCUDAModule(Module &M) {
    return M.getFunction("__cuda_register_globals") != nullptr;
  }

  // Add a call to __proteus_cudart_builtins_init to the global constructors of
  // the module to initialize the Proteus CUDA runtime builtins.
  void emitProteusCUDARuntimeBuiltinsInit(Module &M) {
    FunctionCallee InitFn =
        M.getOrInsertFunction("__proteus_cudart_builtins_init",
                              FunctionType::get(Types.VoidTy, false));

    appendToGlobalCtors(M, cast<Function>(InitFn.getCallee()), 65535);
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
};

// New PM implementation.
struct LambdaPass : PassInfoMixin<LambdaPass> {
  LambdaPass(bool IsLTO) : IsLTO(IsLTO) {}
  bool IsLTO;

  PreservedAnalyses run(Module &M, ModuleAnalysisManager & /*AM*/) {
    LambdaPassImpl PPI{M};

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
struct LegacyLambdaPass : public ModulePass {
  static char ID;
  LegacyLambdaPass() : ModulePass(ID) {}
  bool runOnModule(Module &M) override {
    LambdaPassImpl PPI{M};
    bool Changed = PPI.run(M, false);
    return Changed;
  }
};
} // namespace

//-----------------------------------------------------------------------------
// New PM Registration
//-----------------------------------------------------------------------------
llvm::PassPluginLibraryInfo getLambdaPassPluginInfo() {
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
  // PM.registerPipelineEarlySimplificationEPCallback
#if LLVM_VERSION_MAJOR >= 20
    PB.registerPipelineStartEPCallback([&](ModulePassManager &MPM,
                                           OptimizationLevel,
                                           ThinOrFullLTOPhase LTOPhase) {
      if (LTOPhase != ThinOrFullLTOPhase::None) {
        reportFatalError("Expected registration only for non-LTO");
      }
#else
    PB.registerPipelineStartEPCallback(
        [&](ModulePassManager &MPM, OptimizationLevel) {
#endif
      MPM.addPass(LambdaPass{false});
      return true;
    });

    PB.registerFullLinkTimeOptimizationEarlyEPCallback(
        [&](ModulePassManager &MPM, auto) {
          MPM.addPass(LambdaPass{true});
          return true;
        });
  };

  return {LLVM_PLUGIN_API_VERSION, "LambdaPass", LLVM_VERSION_STRING, Callback};
}

// This is the core interface for pass plugins. It guarantees that 'opt' will
// be able to recognize LambdaPass when added to the pass pipeline on the
// command line, i.e. via '-passes=lambda-pass'
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getLambdaPassPluginInfo();
}

//-----------------------------------------------------------------------------
// Legacy PM Registration
//-----------------------------------------------------------------------------
// The address of this variable is used to uniquely identify the pass. The
// actual value doesn't matter.
char LegacyLambdaPass::ID = 0;

// This is the core interface for pass plugins. It guarantees that 'opt' will
// recognize LegacyLambdaPass when added to the pass pipeline on the command
// line, i.e.  via '--legacy-lambda-pass'
static RegisterPass<LegacyLambdaPass>
    X("legacy-lambda-pass", "Lambda Pass",
      false, // This pass doesn't modify the CFG => false
      false  // This pass is not a pure analysis pass => false
    );
