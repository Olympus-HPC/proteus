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
    // We need collect any kernel host stubs to pass to parse annotations, used
    // in forced annotations.
    const auto StubToKernelMap = getKernelHostStubs(M);

    // if (isDeviceCompilation(M)) {
    //   llvm::outs() << "Device Module\n";
    //   llvm::outs() << M;
    //   llvm::outs() << "End device Module\n";
    // }
    // llvm::outs() << "Host Module\n";
    // llvm::outs() << M;
    // llvm::outs() << "End host Module\n";
    // Lambda analysis pipeline
    // (1) instrumentJitVariableStructIndex analyzes jit_variable calls and
    // marks which indices of a
    //     given lambda class have been designated as runtime constants
    // (2) We then instrument __jit_push_lambda_runtime_constant into device
    // stubs in the case of GPU/CPU
    //.    compilation, or before the callsite of the lambda operator in the
    //case of host execution
    DenseMap<StructType *, SmallVector<LambdaJitVariableInfo, 16>>
        LambdaStorageTypeToJitIndices;

    DenseMap<Type *, GlobalVariable *> LambdaTypeToGlobalName;
    StringMap<Type *> LambdaGlobalNameToType;

    registerLambdaFunctions(M);
    registerJitVariablesWithLambda(M, LambdaTypeToGlobalName);
    for (auto [Ty, NameVar] : LambdaTypeToGlobalName) {
      LambdaGlobalNameToType[getGlobalString(NameVar).value()] = Ty;
      llvm::outs() << getGlobalString(NameVar).value() << "\n";
    }
    instrumentJitVariableStructIndex(M, LambdaTypeToGlobalName,
                                     LambdaStorageTypeToJitIndices);
    if (StubToKernelMap.size())
      instrumentJitPushCalls(M, LambdaStorageTypeToJitIndices,
                             LambdaTypeToGlobalName, LambdaGlobalNameToType,
                             StubToKernelMap);


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
        StubToKernelMap[Key] = GV;
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

  /// This function tells the Proteus runtime which variables to replace with
  /// constants at runtime within a given lambda.  Here's how it works:
  /// 1. Start a use-def analysis at the callbase of each jit_variable
  /// function
  /// 2. Do very simple use-def traversal to find the associated
  /// anonymous class (e.g. class.anon)
  /// 3. Look at all callbases of each proteus::register_lambda template
  /// instantiation. Because we force passage of the lambda by value to
  /// register_lambda, the instantiation must contain an AllocaInst of the
  /// lambda's corresponding anonymous class.  The demangled name of the
  /// lambda can be deduced from the name of the Clang-generated template
  /// instantiation.
  /// 4. Inject the demangled name into the original callbase of the
  /// `jit_variable` function.
  void registerJitVariablesWithLambda(
      Module &M, DenseMap<Type *, GlobalVariable *> &LambdaTypeToGlobalName) {
    llvm::SmallVector<CallBase *, 16> JitVarCallsites;
    llvm::DenseMap<CallBase *, Type *> CallBaseToLambda;
    for (auto &F : M.getFunctionList()) {
      std::string DemangledName = demangle(F.getName().str());
      if (StringRef{DemangledName}.contains("proteus::jit_variable")) {
        for (User *Usr : F.users()) {
          CallBase *CB = dyn_cast<CallBase>(Usr);
          if (!CB)
            continue;
          JitVarCallsites.push_back(CB);
        }
      }
    }

    for (CallBase *CB : JitVarCallsites) {
      // traverse use-def from each callsite, the CB value will be used by the
      // allocated lambda class.
      std::queue<Value *> WorkList;
      llvm::DenseSet<Value *> Discovered;
      WorkList.push(CB);
      for (User *Usr : CB->users()) {
        WorkList.push(Usr);
      }
      while (!WorkList.empty()) {
        Value *Val = WorkList.front();
        WorkList.pop();
        if (Discovered.contains(Val))
          continue;
        Discovered.insert(Val);
        if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Val)) {
          // gep
          WorkList.push(GEP->getPointerOperand());
        } else if (StoreInst *Store = dyn_cast<StoreInst>(Val)) {
          // store. E.G. to the first field of a lambda class
          WorkList.push(Store->getPointerOperand());
        } else if (AllocaInst *Alloc = dyn_cast<AllocaInst>(Val)) {
          // alloca
          StructType *LambdaType =
              dyn_cast<StructType>(Alloc->getAllocatedType());
          if (!LambdaType)
            continue;
          // We found the allocation site for the lambda
          CallBaseToLambda[CB] = LambdaType;
          break;
        }
      }
    }
    IRBuilder<> IRB{M.getContext()};

    // Create a global variable for each lambda type registered
    for (auto &F : M.getFunctionList()) {
      if (!StringRef{demangle(F.getName().str())}.contains(
              "proteus::register_lambda"))
        continue;
      // Alloca must be in the entry block.
      AllocaInst *LambdaAlloca = nullptr;
      for (Instruction &I : F.getEntryBlock()) {
        // By our definition, we force register_lambda to allocate a copy
        // of the lambda struct in our entry block
        auto *Alloc = dyn_cast<AllocaInst>(&I);
        if (!Alloc)
          continue;
        StructType *StructTy = dyn_cast<StructType>(Alloc->getAllocatedType());
        if (!StructTy)
          continue;
        if (LambdaAlloca)
          reportFatalError("Error in LLVM IR of "
                           "proteus::register_lambda--found multiple alloca");
        // We found the allocation site of the lambda.
        LambdaAlloca = Alloc;
        std::string DemangledFuncName = demangle(F.getName().str());
        StringRef DemangledLambdaName =
            parseLambdaType(DemangledFuncName, "proteus::register_lambda");
        LambdaTypeToGlobalName[Alloc->getAllocatedType()] =
            IRB.CreateGlobalString(DemangledLambdaName, ".str", 0, &M);
      }
      if (!LambdaAlloca)
        reportFatalError("Error in LLVM IR of proteus::register_lambda--no "
                         "lambda alloca site found");
    }

    // Inject lambda's Clang-generated name into the jit_variable callsite.
    for (auto &[CB, LambdaType] : CallBaseToLambda) {
      auto It = LambdaTypeToGlobalName.find(LambdaType);
      if (It == LambdaTypeToGlobalName.end())
        reportFatalError("Failed to find the lambda association info");
      CB->setArgOperand(3, It->second);
    }
  }

  FunctionCallee getJitPushLambdaRuntimeConstant(Module &M) {
    // __jit_register_variable_instance_raw(
    //   int32_t Type, int32_t Pos, int32_t Offset,
    //   void const* ValuePtr, int32_t Size,
    //   char const* LambdaType, void const* Key)
    FunctionType *FnTy = FunctionType::get(
        Types.VoidTy,
        {Types.Int32Ty, Types.Int32Ty, Types.Int32Ty, Types.PtrTy, Types.PtrTy},
        /*isVarArg=*/false);
    return M.getOrInsertFunction("__jit_push_lambda_runtime_constant", FnTy);
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
      Module &M,
      DenseMap<Type *, GlobalVariable *> &RegisteredLambdaStorageTypes,
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
      Type *LoadType = F->getReturnType();
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
        llvm::outs() << "ITER: " << *CurVal << '\n';
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
              !RegisteredLambdaStorageTypes.contains(PossibleLamType))
            continue;
          ++numSlotsFound;
          JitIndices[PossibleLamType].emplace_back(
              LambdaJitVariableInfo{Slot, Slot, LoadType});
          if (!LoadType)
            reportFatalError("Load type is null\n");
        }
        if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(CurVal)) {
          StructType *PossibleLamTyp =
              dyn_cast<StructType>(GEP->getSourceElementType());
          if (!PossibleLamTyp ||
              !RegisteredLambdaStorageTypes.contains(PossibleLamTyp))
            continue;
          auto *Slot = GEP->getOperand(GEP->getNumOperands() - 1);
          const StructLayout *SL =
              M.getDataLayout().getStructLayout(PossibleLamTyp);
          ConstantInt *SlotC = dyn_cast<ConstantInt>(Slot);
          if (!SlotC)
            reportFatalError("Expected constant slot");

          auto Offset = SL->getElementOffset(SlotC->getZExtValue());
          Constant *OffsetCI = ConstantInt::get(Types.Int32Ty, Offset);
          // Type* LoadType = GEP->getAccessType();
          JitIndices[PossibleLamTyp].emplace_back(
              LambdaJitVariableInfo{Slot, OffsetCI, LoadType});
          ++numSlotsFound;
          // break;
        }
      }
      llvm::outs() << "numvars = " << numJitVars << "\n";
      llvm::outs() << "numslots found = " << numSlotsFound << "\n";
      if (numJitVars != numSlotsFound)
        reportFatalError("Expected number of jit variables callsites to equal "
                         "the number of slots found");
    }
  }

  void instrumentJitPushCalls(
      Module &M,
      const DenseMap<StructType *, SmallVector<LambdaJitVariableInfo, 16>>
          &LambdaStorageTypeToJitIndices,
      const DenseMap<Type *, GlobalVariable *> &LambdaTypeToGlobalName,
      StringMap<Type *> &LambdaGlobalNameToType,
      const DenseMap<Value *, GlobalVariable *> &StubToKernelMap) {
    Function *LaunchKernelFn = nullptr;
    if (!LaunchFunctionName) {
      reportFatalError("expected launch function name");
    }
    LaunchKernelFn = M.getFunction(LaunchFunctionName);
    for (Value *V : LaunchKernelFn->users()) {
      CallBase *LaunchKernelCB = dyn_cast<CallBase>(V);
      if (!LaunchKernelCB)
        continue;
      Function *Caller = LaunchKernelCB->getFunction();
      if (!StubToKernelMap.contains(Caller))
        continue;
      auto demangledName = demangle(Caller->getName().str());
      llvm::errs() << "DEMANGELD NAME = " << demangledName << "\n";
      auto lambdaName = parseLambdaType(demangledName, "__device_stub__kernel");
      llvm::errs() << "keyword : " << lambdaName << "\n";
      llvm::errs().flush();
      auto it = LambdaGlobalNameToType.find(lambdaName);
      llvm::errs() << "SEARCHING FOR\n";
      for (auto &[Name, Ty] : LambdaGlobalNameToType) {
        llvm::errs() << Name << " : " << *Ty << "\n";
      }
      llvm::errs().flush();
      if (it == LambdaGlobalNameToType.end())
        continue;
      auto *lambdaType = it->second;
      StructType *LambdaStorageClass = dyn_cast<StructType>(lambdaType);
      if (!LambdaStorageClass)
        reportFatalError(
            "Lambda storage class not found.  Internal proteus error.");
      Value *LambdaStoragePtr = nullptr;
      size_t numArgs = LaunchKernelCB->arg_size();
      if (numArgs < 3)
        reportFatalError("hipLaunchKernel call has too few args");
      auto *argPack = LaunchKernelCB->getArgOperand(numArgs - 3);

      auto matchesLambdaStoragePtr = [&](Value *V) -> bool {
        if (!V)
          return false;

        Value *Obj =
            getUnderlyingObject(V->stripPointerCasts(), /*MaxLookup=*/6);
        if (!Obj)
          return false;
        Obj = Obj->stripPointerCasts();

        if (auto *A = dyn_cast<Argument>(Obj)) {
          if (!A->hasByValAttr())
            return false;
          return A->getParamByValType() == LambdaStorageClass;
        }

        if (auto *AI = dyn_cast<AllocaInst>(Obj))
          return AI->getAllocatedType() == LambdaStorageClass;

        return false;
      };

      DenseMap<GetElementPtrInst *, SmallVector<StoreInst *>> GEPsToKernelArgs;
      SmallVector<Value *, 16> ArgPackWorkList;
      DenseSet<Value *> SeenArgPackVals;
      ArgPackWorkList.push_back(argPack);
      while (!ArgPackWorkList.empty()) {
        Value *Cur = ArgPackWorkList.pop_back_val();
        if (SeenArgPackVals.contains(Cur))
          continue;
        SeenArgPackVals.insert(Cur);

        for (auto *Usr : argPack->users()) {
          if (auto *GEP = dyn_cast<GetElementPtrInst>(Usr)) {
            for (auto *GEPUser : GEP->users())
              if (StoreInst *Store = dyn_cast<StoreInst>(GEPUser))
                GEPsToKernelArgs[GEP].emplace_back(Store);
            continue;
          }
          if (isa<BitCastInst>(Usr) || isa<AddrSpaceCastInst>(Usr))
            ArgPackWorkList.push_back(Usr);
        }
      }

      Instruction *LaunchI = dyn_cast<Instruction>(LaunchKernelCB);
      if (!LaunchI)
        reportFatalError("expected hipLaunchKernel call to be an Instruction");

      DominatorTree DT(*Caller);
      StoreInst *BestStore = nullptr;
      unsigned BestLevel = 0;
      for (auto &[_, StoreInstVec] : GEPsToKernelArgs) {
        for (StoreInst *SI : StoreInstVec) {
          if (!DT.dominates(SI, LaunchI))
            continue;
          if (!matchesLambdaStoragePtr(SI->getValueOperand()))
            continue;

          unsigned Level = DT.getNode(SI->getParent())->getLevel();
          if (!BestStore || Level > BestLevel ||
              (Level == BestLevel &&
               BestStore->getParent() == LaunchI->getParent() &&
               BestStore->comesBefore(SI))) {
            BestStore = SI;
            BestLevel = Level;
          }
        }
      }

      if (BestStore)
        LambdaStoragePtr = BestStore->getValueOperand();
      // Check the args for the storage class
      if (!LambdaStoragePtr) {
        for (Argument &A : Caller->args()) {
          if (A.hasByValAttr() && A.getParamByValType() == LambdaStorageClass) {
            LambdaStoragePtr = &A;
            break;
          }
        }
      }

      if (!LambdaStoragePtr)
        reportFatalError(
            "Failed to locate lambda storage pointer for hipLaunchKernel");

      // inject the call to our runtime before the kernel launch
      IRBuilder<> Builder(LaunchKernelCB);
      auto JitVarFn = getJitPushLambdaRuntimeConstant(M);

      for (auto JitVarInfo :
           LambdaStorageTypeToJitIndices.find(LambdaStorageClass)
               ->getSecond()) {
        if (!JitVarInfo.Type)
          reportFatalError("jit var info type is null\n");
        auto ProteusRCType = getRCTypeForLLVMType(JitVarInfo.Type);
        ConstantInt *SlotC = dyn_cast<ConstantInt>(JitVarInfo.Slot);

        if (!SlotC)
          reportFatalError("Expected constant slot");
        auto Idx = SlotC->getZExtValue();
        if (!ProteusRCType)
          reportFatalError("Proteus does not support user-specified type as a "
                           "jit::variable");
        Value *FieldPtr =
            Builder.CreateStructGEP(LambdaStorageClass, LambdaStoragePtr, Idx);
        Builder.CreateCall(
            JitVarFn,
            {Builder.getInt32(static_cast<int32_t>(ProteusRCType.value())),
             Builder.getInt32(Idx), JitVarInfo.Offset, FieldPtr,
             LambdaTypeToGlobalName.find(LambdaStorageClass)->second});
      }
    }
  }

  // Parse the lambda name from the template parameters of a function template
  // E.G. given "Haystack<Needle>" as DemangledName and "Haystack" as
  // FuncTemplateName return "Needle"
  StringRef parseLambdaType(StringRef DemangledName,
                            const char *FuncTemplateName) {
    int L = -1;
    int R = -1;
    int Level = 0;
    // Start after the function symbol to avoid parsing its templated return
    // type.
    size_t Start = DemangledName.find(FuncTemplateName);
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
  }

  void registerLambdaFunctions(Module &M) {
    DEBUG(Logger::logs("lambda-pass")
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
      StringRef LambdaType =
          parseLambdaType(DemangledName, "proteus::register_lambda");

      DEBUG(Logger::logs("lambda-pass")
            << Function->getName() << " " << DemangledName << " " << LambdaType
            << "\n");
      llvm::outs() << "Demangled name = " << DemangledName << "lambda type "
                   << LambdaType << "\n";
      for (auto *User : Function->users()) {
        CallBase *CB = dyn_cast<CallBase>(User);
        if (!CB)
          reportFatalError("Expected CallBase as user of "
                           "proteus::register_lambda function");

        IRBuilder<> Builder(CB);
        auto *LambdaNameGlobal = Builder.CreateGlobalString(LambdaType);
        // Sometimes, whenever a function returns a struct, clang will
        // automatically convert one of the arguments into holding the struct
        // return pointer. We need to modify the last argoperand of the
        // register_lambda call so we check if we have an sret argument
        bool HasSRETArg = false;
        for (uint32_t I = 0; I < CB->getNumOperands(); ++I) {
          HasSRETArg =
              HasSRETArg || CB->paramHasAttr(I, llvm::Attribute::StructRet);
        }
        int LambdaNameIndex = HasSRETArg ? 2 : 1;
        CB->setArgOperand(LambdaNameIndex, LambdaNameGlobal);
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
