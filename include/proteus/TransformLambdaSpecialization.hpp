//===-- TransformLambdaSpecialization.hpp -- Specialize arguments --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_TRANSFORM_LAMBDA_SPECIALIZATION_HPP
#define PROTEUS_TRANSFORM_LAMBDA_SPECIALIZATION_HPP

#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/Debug.h"
#include "proteus/Utils.h"

#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>

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

  static void handleLoad(Module &M, User *User,
                         const SmallVector<RuntimeConstant> &RCVec) {
    auto *Arg = findArgByPos(RCVec, 0);
    if (!Arg)
      return;

    Constant *C = getConstant(M.getContext(), User->getType(), *Arg);
    User->replaceAllUsesWith(C);
    PROTEUS_DBG(Logger::logs("proteus") << traceOut(Arg->Pos, C));
    if (Config::get().ProteusTraceOutput >= 1)
      Logger::trace(traceOut(Arg->Pos, C));
  }

  static void handleGEP(Module &M, GetElementPtrInst *GEP, User *User,
                        const SmallVector<RuntimeConstant> &RCVec) {
    auto *GEPSlot = GEP->getOperand(User->getNumOperands() - 1);
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
      if (Config::get().ProteusTraceOutput >= 1)
        Logger::trace(traceOut(Arg->Pos, C));
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

    PROTEUS_DBG(Logger::logs("proteus") << "\t users" << "\n");
    for (User *User : LambdaClass->users()) {
      PROTEUS_DBG(Logger::logs("proteus") << *User << "\n");
      if (isa<LoadInst>(User))
        handleLoad(M, User, RCVec);
      else if (auto *GEP = dyn_cast<GetElementPtrInst>(User))
        handleGEP(M, GEP, User, RCVec);
    }
  }
};

} // namespace proteus

#endif
