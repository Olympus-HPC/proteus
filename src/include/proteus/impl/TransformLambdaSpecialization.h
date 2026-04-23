//===-- TransformLambdaSpecialization.h -- Specialize arguments --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_TRANSFORM_LAMBDA_SPECIALIZATION_H
#define PROTEUS_TRANSFORM_LAMBDA_SPECIALIZATION_H

#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/impl/CoreLLVM.h"
#include "proteus/impl/Debug.h"
#include "proteus/impl/Utils.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <llvm/Transforms/Utils/Cloning.h>

#include <cstdint>
#include <cstring>

namespace proteus {

using namespace llvm;

inline Constant *getConstant(LLVMContext &Ctx, Type *ArgType,
                             const RuntimeConstant &RC) {
  switch (RC.Type) {
  case RuntimeConstantType::BOOL:
    return ConstantInt::get(ArgType, RC.Value.BoolVal);
  case RuntimeConstantType::INT8:
    return ConstantInt::get(ArgType, RC.Value.Int8Val);
  case RuntimeConstantType::INT32:
    return ConstantInt::get(ArgType, RC.Value.Int32Val);
  case RuntimeConstantType::INT64:
    return ConstantInt::get(ArgType, RC.Value.Int64Val);
  case RuntimeConstantType::FLOAT:
    return ConstantFP::get(ArgType, RC.Value.FloatVal);
  case RuntimeConstantType::DOUBLE:
    return ConstantFP::get(ArgType, RC.Value.DoubleVal);
  case RuntimeConstantType::LONG_DOUBLE:
    return ConstantFP::get(ArgType, RC.Value.LongDoubleVal);
  case RuntimeConstantType::PTR: {
    auto *IntC = ConstantInt::get(Type::getInt64Ty(Ctx), RC.Value.Int64Val);
    return ConstantExpr::getIntToPtr(IntC, ArgType);
  }
  default:
    std::string TypeString;
    raw_string_ostream TypeOstream(TypeString);
    ArgType->print(TypeOstream);
    reportFatalError("JIT Incompatible type in runtime constant: " +
                     TypeOstream.str());
  }
}

class TransformLambdaSpecialization {
private:
  using JitVariantMap = DenseMap<int32_t, RuntimeConstant>;

  static const RuntimeConstant *
  findArgByOffset(const JitVariantMap &RCMap, int32_t Offset) {
    for (auto &[_, Arg] : RCMap) {
      if (Arg.Offset == Offset)
        return &Arg;
    }
    return nullptr;
  };

  static const RuntimeConstant *
  findArgByPos(const JitVariantMap &RCMap, int32_t Pos) {
    auto It = RCMap.find(Pos);
    if (It == RCMap.end())
      return nullptr;
    return &It->second;
  };

  static auto traceOut(int Slot, Constant *C) {
    SmallString<128> S;
    raw_svector_ostream OS(S);
    OS << "[LambdaSpec] Replacing slot " << Slot << " with " << *C << "\n";

    return S;
  };

  static void handleLoad(Module &M, LoadInst *LI, const JitVariantMap &RCVec) {
    auto *Arg = findArgByPos(RCVec, 0);
    if (!Arg)
      return;

    Constant *C = getConstant(M.getContext(), LI->getType(), *Arg);
    LI->replaceAllUsesWith(C);
    PROTEUS_DBG(Logger::logs("proteus") << traceOut(Arg->Pos, C));
    if (Config::get().traceSpecializations())
      Logger::trace(traceOut(Arg->Pos, C));
  }

  static void handleGEP(Module &M, GetElementPtrInst *GEP,
                        const JitVariantMap &RCVec) {
    auto *GEPSlot = GEP->getOperand(GEP->getNumOperands() - 1);
    ConstantInt *CI = dyn_cast<ConstantInt>(GEPSlot);
    int Slot = CI->getZExtValue();
    Type *SrcTy = GEP->getSourceElementType();

    auto *Arg = SrcTy->isStructTy() ? findArgByPos(RCVec, Slot)
                                    : findArgByOffset(RCVec, Slot);
    if (!Arg)
      return;

    for (auto *GEPUser : GEP->users()) {
      auto *LI = dyn_cast<LoadInst>(GEPUser);
      if (!LI)
        reportFatalError("Expected load instruction");
      Type *LoadType = LI->getType();
      Constant *C = getConstant(M.getContext(), LoadType, *Arg);
      LI->replaceAllUsesWith(C);
      PROTEUS_DBG(Logger::logs("proteus") << traceOut(Arg->Pos, C));
      if (Config::get().traceSpecializations())
        Logger::trace(traceOut(Arg->Pos, C));
    }
  }

  static Function *findLambdaOperatorForFunctor(Module &M, uint64_t FunctorID) {
    SmallVector<std::pair<Function *, uint64_t>> LambdaOperators;
    findFunctionsWithU64Metadata(M, "proteus.registered_lambda", LambdaOperators);
    for (auto [Lambda, ID] : LambdaOperators)
      if (ID == FunctorID)
        return Lambda;
    return nullptr;
  }

  static CallBase *findDirectCallTo(Function &Caller, const Function &Callee) {
    for (BasicBlock &BB : Caller) {
      for (Instruction &I : BB) {
        auto *CB = dyn_cast<CallBase>(&I);
        if (!CB)
          continue;
        Function *Called = CB->getCalledFunction();
        if (!Called)
          continue;
        if (Called == &Callee)
          return CB;
      }
    }
    return nullptr;
  }

  static Function *cloneForVariant(Function &F, uint64_t FunctorID,
                                   size_t VariantIndex) {
    ValueToValueMapTy VMap;

    auto *NewFunc = Function::Create(
        F.getFunctionType(), F.getLinkage(), F.getAddressSpace(),
        F.getName() + ".proteus.variant." + Twine(FunctorID) + "." +
            Twine(VariantIndex),
        F.getParent());

    NewFunc->copyAttributesFrom(&F);
    NewFunc->setCallingConv(F.getCallingConv());

    auto NewArgIt = NewFunc->arg_begin();
    for (auto &OldArg : F.args()) {
      NewArgIt->setName(OldArg.getName());
      VMap[&OldArg] = &(*NewArgIt++);
    }

    SmallVector<ReturnInst *, 8> Returns;
    CloneFunctionInto(NewFunc, &F, VMap,
                      CloneFunctionChangeType::LocalChangesOnly, Returns);

    // The original lambda operator is marked noinline to preserve a distinct
    // call target. Allow the specialized clones to be inlined by later O3.
    NewFunc->removeFnAttr(Attribute::NoInline);

    return NewFunc;
  }

  static StructType *inferStructTypeFromGEPValue(Value *Ptr) {
    auto *GEP = dyn_cast<GetElementPtrInst>(Ptr);
    if (!GEP)
      return nullptr;

    Type *CurTy = GEP->getSourceElementType();
    bool IsFirstIndex = true;
    for (Value *Idx : GEP->indices()) {
      // The first GEP index selects an element of the "source element" type
      // as an array; it does not descend into a struct field.
      if (IsFirstIndex) {
        IsFirstIndex = false;
        continue;
      }

      auto *CI = dyn_cast<ConstantInt>(Idx);
      if (!CI)
        return nullptr;

      if (auto *ST = dyn_cast<StructType>(CurTy)) {
        uint64_t ElemIdx = CI->getZExtValue();
        if (ElemIdx >= ST->getNumElements())
          return nullptr;
        CurTy = ST->getElementType(ElemIdx);
        continue;
      }

      if (auto *AT = dyn_cast<ArrayType>(CurTy)) {
        CurTy = AT->getElementType();
        continue;
      }

      if (auto *VT = dyn_cast<VectorType>(CurTy)) {
        CurTy = VT->getElementType();
        continue;
      }

      return nullptr;
    }

    return dyn_cast<StructType>(CurTy);
  }

  static Value *buildSingleCompare(IRBuilder<> &B, const DataLayout &DL,
                                  Value *LambdaObjPtr, StructType *LambdaTy,
                                  const RuntimeConstant &RC) {
    Value *FieldPtr = nullptr;
    if (LambdaTy && RC.Pos >= 0 &&
        static_cast<uint64_t>(RC.Pos) < LambdaTy->getNumElements()) {
      FieldPtr = B.CreateStructGEP(LambdaTy, LambdaObjPtr, RC.Pos);
    } else {
      if (RC.Offset < 0)
        return nullptr;
      FieldPtr = B.CreateGEP(B.getInt8Ty(), LambdaObjPtr,
                             B.getInt64(static_cast<uint64_t>(RC.Offset)));
    }

    switch (RC.Type) {
    case RuntimeConstantType::BOOL: {
      Value *Loaded = B.CreateAlignedLoad(B.getInt8Ty(), FieldPtr, Align(1));
      Value *C = B.getInt8(RC.Value.BoolVal ? 1 : 0);
      return B.CreateICmpEQ(Loaded, C);
    }
    case RuntimeConstantType::INT8: {
      Value *Loaded = B.CreateAlignedLoad(B.getInt8Ty(), FieldPtr, Align(1));
      Value *C = B.getInt8(static_cast<uint8_t>(RC.Value.Int8Val));
      return B.CreateICmpEQ(Loaded, C);
    }
    case RuntimeConstantType::INT32: {
      Value *Loaded = B.CreateAlignedLoad(B.getInt32Ty(), FieldPtr, Align(1));
      Value *C = B.getInt32(static_cast<uint32_t>(RC.Value.Int32Val));
      return B.CreateICmpEQ(Loaded, C);
    }
    case RuntimeConstantType::INT64: {
      Value *Loaded = B.CreateAlignedLoad(B.getInt64Ty(), FieldPtr, Align(1));
      Value *C = B.getInt64(static_cast<uint64_t>(RC.Value.Int64Val));
      return B.CreateICmpEQ(Loaded, C);
    }
    case RuntimeConstantType::FLOAT: {
      Value *LoadedF = B.CreateAlignedLoad(B.getFloatTy(), FieldPtr, Align(1));
      Value *LoadedBits = B.CreateBitCast(LoadedF, B.getInt32Ty());
      uint32_t Bits = 0;
      std::memcpy(&Bits, &RC.Value.FloatVal, sizeof(Bits));
      Value *C = B.getInt32(Bits);
      return B.CreateICmpEQ(LoadedBits, C);
    }
    case RuntimeConstantType::DOUBLE: {
      Value *LoadedF =
          B.CreateAlignedLoad(B.getDoubleTy(), FieldPtr, Align(1));
      Value *LoadedBits = B.CreateBitCast(LoadedF, B.getInt64Ty());
      uint64_t Bits = 0;
      std::memcpy(&Bits, &RC.Value.DoubleVal, sizeof(Bits));
      Value *C = B.getInt64(Bits);
      return B.CreateICmpEQ(LoadedBits, C);
    }
    case RuntimeConstantType::PTR: {
      Type *IntPtrTy = DL.getIntPtrType(B.getContext());
      Value *Loaded = B.CreateAlignedLoad(IntPtrTy, FieldPtr, Align(1));
      uint64_t Bits =
          static_cast<uint64_t>(reinterpret_cast<uintptr_t>(RC.Value.PtrVal));
      Value *C = ConstantInt::get(IntPtrTy, Bits);
      return B.CreateICmpEQ(Loaded, C);
    }
    default:
      return nullptr;
    }
  }

  static Value *buildVariantMatch(IRBuilder<> &B, const DataLayout &DL,
                                 Value *LambdaObjPtr, StructType *LambdaTy,
                                 const JitVariantMap &Variant) {
    if (Variant.empty())
      return B.getInt1(true);

    SmallVector<int32_t, 16> Keys;
    Keys.reserve(Variant.size());
    for (const auto &KV : Variant)
      Keys.push_back(KV.first);

    llvm::sort(Keys.begin(), Keys.end());

    Value *Match = B.getInt1(true);
    for (int32_t K : Keys) {
      auto It = Variant.find(K);
      if (It == Variant.end())
        reportFatalError("Internal error: DenseMap key vanished during match");
      Value *Cmp =
          buildSingleCompare(B, DL, LambdaObjPtr, LambdaTy, It->second);
      if (!Cmp)
        return nullptr;
      Match = B.CreateAnd(Match, Cmp);
    }

    return Match;
  }

  static void specializeCallOperator(Module &M, Function &CallOp,
                                     const JitVariantMap &RCMap) {
    if (CallOp.arg_empty())
      return;

    auto *LambdaClass = CallOp.getArg(0);
    PROTEUS_DBG(Logger::logs("proteus")
                << "[LambdaSpec] Function: " << CallOp.getName()
                << " RCVec size " << RCMap.size() << "\n");

    for (User *U : LambdaClass->users()) {
      if (auto *LI = dyn_cast<LoadInst>(U))
        handleLoad(M, LI, RCMap);
      else if (auto *GEP = dyn_cast<GetElementPtrInst>(U))
        handleGEP(M, GEP, RCMap);
    }
  }

public:
  static void transform(Module &M, Function &FunctorOperatorFunction,
                        uint64_t FunctorID,
                        ArrayRef<JitVariantMap> Variants) {
    if (Variants.empty())
      return;

    Function *LambdaOperatorMethod = findLambdaOperatorForFunctor(M, FunctorID);
    if (!LambdaOperatorMethod) {
      // On host (and sometimes device) the lambda call operator may be fully
      // inlined into the functor wrapper call operator, leaving no separate
      // `lambda::operator()` function to tag.
      if (Variants.size() == 1)
        specializeCallOperator(M, FunctorOperatorFunction, Variants.front());
      return;
    }

    if (Variants.size() == 1) {
      specializeCallOperator(M, *LambdaOperatorMethod, Variants.front());
      return;
    }

    CallBase *OrigCall =
        findDirectCallTo(FunctorOperatorFunction, *LambdaOperatorMethod);
    if (!OrigCall)
      return;

    CallingConv::ID CallConv = OrigCall->getCallingConv();
    AttributeList CallAttrs = OrigCall->getAttributes();

    SmallVector<Value *, 8> CallArgs;
    CallArgs.reserve(OrigCall->arg_size());
    for (Use &U : OrigCall->args())
      CallArgs.push_back(U.get());

    Value *LambdaObjPtr = OrigCall->getArgOperand(0);
    StructType *LambdaTy = inferStructTypeFromGEPValue(LambdaObjPtr);

    // Split out the original call (and everything after it) into a tail block.
    BasicBlock *EntryBB = OrigCall->getParent();
    BasicBlock *TailBB = EntryBB->splitBasicBlock(OrigCall, "proteus.after_call");

    // Remove the unconditional branch produced by splitBasicBlock.
    EntryBB->getTerminator()->eraseFromParent();

    // Create a PHI in the tail block for any call result.
    PHINode *ResultPhi = nullptr;
    Type *RetTy = OrigCall->getType();
    if (!RetTy->isVoidTy()) {
      ResultPhi = PHINode::Create(RetTy, Variants.size() + 1,
                                  "proteus.lambda_dispatch.result",
                                  &*TailBB->begin());
      OrigCall->replaceAllUsesWith(ResultPhi);
    }

    Instruction *AfterCallIP = OrigCall->getNextNode();
    OrigCall->eraseFromParent();

    Function *Wrapper = &FunctorOperatorFunction;
    LLVMContext &Ctx = M.getContext();

    BasicBlock *FallbackBB =
        BasicBlock::Create(Ctx, "proteus.lambda_dispatch.fallback", Wrapper);

    // Build the chain of check blocks (entry -> check0 -> ... -> fallback).
    BasicBlock *CurCheckBB = EntryBB;
    for (size_t I = 0, E = Variants.size(); I < E; ++I) {
      BasicBlock *VariantBB = BasicBlock::Create(
          Ctx, "proteus.lambda_dispatch.variant." + Twine(I), Wrapper);
      BasicBlock *NextCheckBB =
          (I + 1 == E)
              ? FallbackBB
              : BasicBlock::Create(Ctx,
                                   "proteus.lambda_dispatch.check." +
                                       Twine(I + 1),
                                   Wrapper);

      {
        IRBuilder<> B(CurCheckBB);
        Value *Match = buildVariantMatch(B, M.getDataLayout(), LambdaObjPtr,
                                         LambdaTy, Variants[I]);
        if (!Match) {
          // Unsupported runtime-constant type for dispatch; fall back to the
          // unspecialized call.
          BranchInst::Create(FallbackBB, CurCheckBB);
        } else {
          BranchInst::Create(VariantBB, NextCheckBB, Match, CurCheckBB);
        }
      }

      // Emit the specialized call in the variant block.
      {
        IRBuilder<> B(VariantBB);
        Function *Clone = cloneForVariant(*LambdaOperatorMethod, FunctorID, I);
        specializeCallOperator(M, *Clone, Variants[I]);

        CallInst *C = B.CreateCall(Clone, CallArgs);
        C->setCallingConv(CallConv);
        C->setAttributes(CallAttrs);

        if (ResultPhi)
          ResultPhi->addIncoming(C, VariantBB);

        B.CreateBr(TailBB);
      }

      CurCheckBB = NextCheckBB;
    }

    // Emit the fallback call to the original, unspecialized operator.
    {
      IRBuilder<> B(FallbackBB);
      CallInst *C = B.CreateCall(LambdaOperatorMethod, CallArgs);
      C->setCallingConv(CallConv);
      C->setAttributes(CallAttrs);

      if (ResultPhi)
        ResultPhi->addIncoming(C, FallbackBB);

      B.CreateBr(TailBB);
    }

    // If the call was immediately followed by a terminator, splitBasicBlock
    // still produced a valid tail block. No further fixup required.
    (void)AfterCallIP;
  }
};

} // namespace proteus

#endif
