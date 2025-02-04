//===-- TransformArgumentSpecialization.hpp -- Specialize arguments --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_TRANSFORM_ARGUMENT_SPECIALIZATION_HPP
#define PROTEUS_TRANSFORM_ARGUMENT_SPECIALIZATION_HPP

#include "llvm/Support/Debug.h"
#include <llvm/IR/IRBuilder.h>

#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/Utils.h"

namespace proteus {

using namespace llvm;

class TransformArgumentSpecialization {
public:
  static void transform(Module &M, Function &F,
                        const SmallVectorImpl<int32_t> &ArgPos,
                        ArrayRef<RuntimeConstant> RC) {
    auto &Ctx = M.getContext();

    // Replace argument uses with runtime constants.
    for (int I = 0; I < ArgPos.size(); ++I) {
      int ArgNo = ArgPos[I];
      Value *Arg = F.getArg(ArgNo);
      Type *ArgType = Arg->getType();
      Constant *C = nullptr;
      if (ArgType->isIntegerTy(1)) {
        C = ConstantInt::get(ArgType, RC[I].Value.BoolVal);
      } else if (ArgType->isIntegerTy(8)) {
        C = ConstantInt::get(ArgType, RC[I].Value.Int8Val);
      } else if (ArgType->isIntegerTy(32)) {
        // Logger::logs("proteus") << "RC is Int32\n";
        C = ConstantInt::get(ArgType, RC[I].Value.Int32Val);
      } else if (ArgType->isIntegerTy(64)) {
        // Logger::logs("proteus") << "RC is Int64\n";
        C = ConstantInt::get(ArgType, RC[I].Value.Int64Val);
      } else if (ArgType->isFloatTy()) {
        // Logger::logs("proteus") << "RC is Float\n";
        C = ConstantFP::get(ArgType, RC[I].Value.FloatVal);
      }
      // NOTE: long double on device should correspond to plain double.
      // XXX: CUDA with a long double SILENTLY fails to create a working
      // kernel in AOT compilation, with or without JIT.
      else if (ArgType->isDoubleTy()) {
        // Logger::logs("proteus") << "RC is Double\n";
        C = ConstantFP::get(ArgType, RC[I].Value.DoubleVal);
      } else if (ArgType->isX86_FP80Ty() || ArgType->isPPC_FP128Ty() ||
                 ArgType->isFP128Ty()) {
        C = ConstantFP::get(ArgType, RC[I].Value.LongDoubleVal);
      } else if (ArgType->isPointerTy()) {
        auto *IntC =
            ConstantInt::get(Type::getInt64Ty(Ctx), RC[I].Value.Int64Val);
        C = ConstantExpr::getIntToPtr(IntC, ArgType);
      } else {
        std::string TypeString;
        raw_string_ostream TypeOstream(TypeString);
        ArgType->print(TypeOstream);
        FATAL_ERROR("JIT Incompatible type in runtime constant: " +
                    TypeOstream.str());
      }

      PROTEUS_DBG(Logger::logs("proteus") << "[ArgSpecial] Replaced Function "
                                          << F.getName() + " ArgNo " << ArgNo
                                          << " with value " << *C << "\n");
      Arg->replaceAllUsesWith(C);
    }
  }
};

} // namespace proteus

#endif
