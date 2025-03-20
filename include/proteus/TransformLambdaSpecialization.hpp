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

#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>

#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/Debug.h"
#include "proteus/Utils.h"

namespace proteus {

using namespace llvm;

inline Constant *getConstant(LLVMContext &Ctx, Type *ArgType,
                             const RuntimeConstant &RC) {
  if (ArgType->isIntegerTy(1)) {
    return ConstantInt::get(ArgType, RC.Value.BoolVal);
  } else if (ArgType->isIntegerTy(8)) {
    return ConstantInt::get(ArgType, RC.Value.Int8Val);
  } else if (ArgType->isIntegerTy(32)) {
    return ConstantInt::get(ArgType, RC.Value.Int32Val);
  } else if (ArgType->isIntegerTy(64)) {
    return ConstantInt::get(ArgType, RC.Value.Int64Val);
  } else if (ArgType->isFloatTy()) {
    return ConstantFP::get(ArgType, RC.Value.FloatVal);
  } else if (ArgType->isDoubleTy()) {
    return ConstantFP::get(ArgType, RC.Value.DoubleVal);
  } else if (ArgType->isX86_FP80Ty() || ArgType->isPPC_FP128Ty() ||
             ArgType->isFP128Ty()) {
    return ConstantFP::get(ArgType, RC.Value.LongDoubleVal);
  } else if (ArgType->isPointerTy()) {
    auto *IntC = ConstantInt::get(Type::getInt64Ty(Ctx), RC.Value.Int64Val);
    return ConstantExpr::getIntToPtr(IntC, ArgType);
  } else {
    std::string TypeString;
    raw_string_ostream TypeOstream(TypeString);
    ArgType->print(TypeOstream);
    PROTEUS_FATAL_ERROR("JIT Incompatible type in runtime constant: " +
                        TypeOstream.str());
  }
}

class TransformLambdaSpecialization {
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
    for (auto &Arg : RCVec) {
      PROTEUS_DBG(Logger::logs("proteus")
                  << "{" << Arg.Value.Int64Val << ", " << Arg.Slot << " }\n");
    }

    PROTEUS_DBG(Logger::logs("proteus") << "\t users" << "\n");
    for (User *User : LambdaClass->users()) {
      PROTEUS_DBG(Logger::logs("proteus") << *User << "\n");
      if (isa<LoadInst>(User)) {
        for (auto &Arg : RCVec) {
          if (Arg.Slot == 0) {
            Constant *C = getConstant(M.getContext(), User->getType(), Arg);
            User->replaceAllUsesWith(C);
            PROTEUS_DBG(Logger::logs("proteus")
                        << "[LambdaSpec] Replacing " << *User << " with " << *C
                        << "\n");
          }
        }
      } else if (auto *GEP = dyn_cast<GetElementPtrInst>(User)) {
        auto *GEPSlot = GEP->getOperand(User->getNumOperands() - 1);
        ConstantInt *CI = dyn_cast<ConstantInt>(GEPSlot);
        int Slot = CI->getZExtValue();
        for (auto &Arg : RCVec) {
          if (Arg.Slot == Slot) {
            for (auto *GEPUser : GEP->users()) {
              auto *LI = dyn_cast<LoadInst>(GEPUser);
              if (!LI)
                PROTEUS_FATAL_ERROR("Expected load instruction");
              Type *LoadType = LI->getType();
              Constant *C = getConstant(M.getContext(), LoadType, Arg);
              LI->replaceAllUsesWith(C);
              PROTEUS_DBG(Logger::logs("proteus")
                          << "[LambdaSpec] Replacing " << *User << " with "
                          << *C << "\n");
            }
          }
        }
      }
    }
  }
};

} // namespace proteus

#endif
