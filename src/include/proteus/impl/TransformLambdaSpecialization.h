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
#include "proteus/impl/Debug.h"
#include "proteus/impl/Utils.h"

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>

#include <limits>
#include <optional>

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
  static const RuntimeConstant *
  findArgByOffset(const SmallVector<RuntimeConstant> &RCVec, int32_t Offset) {
    for (auto &Arg : RCVec) {
      if (Arg.Offset == Offset)
        return &Arg;
    }
    return nullptr;
  };

  static const RuntimeConstant *
  findArgByPos(const SmallVector<RuntimeConstant> &RCVec, int32_t Pos) {
    for (auto &Arg : RCVec) {
      if (Arg.Pos == Pos)
        return &Arg;
    }
    return nullptr;
  };

  static auto traceOut(int Slot, Constant *C) {
    SmallString<128> S;
    raw_svector_ostream OS(S);
    OS << "[LambdaSpec] Replacing slot " << Slot << " with " << *C << "\n";

    return S;
  };

  static void replaceLoad(Module &M, LoadInst *LI, int32_t ByteOffset,
                          const SmallVector<RuntimeConstant> &RCVec) {
    auto *Arg = findArgByOffset(RCVec, ByteOffset);
    if (!Arg && ByteOffset == 0)
      Arg = findArgByPos(RCVec, 0);
    if (!Arg)
      return;

    Constant *C = getConstant(M.getContext(), LI->getType(), *Arg);
    LI->replaceAllUsesWith(C);
    PROTEUS_DBG(Logger::logs("proteus") << traceOut(Arg->Pos, C));
    if (Config::get().traceSpecializations())
      Logger::trace(traceOut(Arg->Pos, C));
  }

  static std::optional<int32_t> getGEPByteOffset(const DataLayout &DL,
                                                 GetElementPtrInst *GEP) {
    APInt Offset(DL.getPointerTypeSizeInBits(GEP->getType()), 0, true);
    if (!GEP->accumulateConstantOffset(DL, Offset))
      return std::nullopt;

    int64_t Offset64 = Offset.getSExtValue();
    if (Offset64 < std::numeric_limits<int32_t>::min() ||
        Offset64 > std::numeric_limits<int32_t>::max())
      return std::nullopt;

    return static_cast<int32_t>(Offset64);
  }

  static void visitPointerUsers(Module &M, Value *Ptr, int32_t ByteOffset,
                                const SmallVector<RuntimeConstant> &RCVec,
                                SmallPtrSetImpl<Value *> &Seen) {
    if (!Seen.insert(Ptr).second)
      return;

    const DataLayout &DL = M.getDataLayout();
    for (User *User : Ptr->users()) {
      if (auto *LI = dyn_cast<LoadInst>(User)) {
        if (LI->getPointerOperand() != Ptr)
          continue;
        replaceLoad(M, LI, ByteOffset, RCVec);
        continue;
      }

      if (auto *GEP = dyn_cast<GetElementPtrInst>(User)) {
        if (GEP->getPointerOperand() != Ptr)
          continue;

        auto LocalOffset = getGEPByteOffset(DL, GEP);
        if (!LocalOffset)
          continue;

        visitPointerUsers(M, GEP, ByteOffset + *LocalOffset, RCVec, Seen);
        continue;
      }

      if (isa<BitCastInst>(User) || isa<AddrSpaceCastInst>(User) ||
          isa<PHINode>(User) || isa<SelectInst>(User)) {
        bool UsesPtr = false;
        for (Value *Operand : User->operands()) {
          if (Operand == Ptr) {
            UsesPtr = true;
            break;
          }
        }
        if (!UsesPtr)
          continue;
        visitPointerUsers(M, cast<Value>(User), ByteOffset, RCVec, Seen);
      }
    }
  }

public:
  static void transform(Module &M, Function &F,
                        const SmallVector<RuntimeConstant> &RCVec) {
    auto *LambdaClass = F.getArg(0);
    PROTEUS_DBG(Logger::logs("proteus")
                << "[LambdaSpec] Function: " << F.getName() << " RCVec size "
                << RCVec.size() << "\n");
    PROTEUS_DBG(Logger::logs("proteus")
                << "TransformLambdaSpecialization::transform" << "\n");
    PROTEUS_DBG(Logger::logs("proteus") << "\t args" << "\n");
    if (Config::get().ProteusDebugOutput) {
      for (auto &Arg : RCVec) {
        Logger::logs("proteus")
            << "{" << Arg.Value.Int64Val << ", " << Arg.Pos << " }\n";
      }
    }

    SmallPtrSet<Value *, 32> Seen;
    visitPointerUsers(M, LambdaClass, /*ByteOffset=*/0, RCVec, Seen);
  }
};

} // namespace proteus

#endif
