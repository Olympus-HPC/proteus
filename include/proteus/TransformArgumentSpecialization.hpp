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

#include <llvm/IR/IRBuilder.h>
#include <llvm/Support/Debug.h>

#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/Debug.h"
#include "proteus/Utils.h"

namespace proteus {

using namespace llvm;

class TransformArgumentSpecialization {
public:
  static void transform(Module &M, Function &F,
                        ArrayRef<RuntimeConstant> RCArray) {
    auto &Ctx = M.getContext();

    // Replace argument uses with runtime constants.
    for (const auto &RC : RCArray) {
      int ArgNo = RC.Pos;
      Value *Arg = F.getArg(ArgNo);
      Type *ArgType = Arg->getType();
      Constant *C = nullptr;

      if (ArgType->isIntegerTy(1)) {
        C = ConstantInt::get(ArgType, RC.Value.BoolVal);
      } else if (ArgType->isIntegerTy(8)) {
        C = ConstantInt::get(ArgType, RC.Value.Int8Val);
      } else if (ArgType->isIntegerTy(32)) {
        // Logger::logs("proteus") << "RC is Int32\n";
        C = ConstantInt::get(ArgType, RC.Value.Int32Val);
      } else if (ArgType->isIntegerTy(64)) {
        // Logger::logs("proteus") << "RC is Int64\n";
        C = ConstantInt::get(ArgType, RC.Value.Int64Val);
      } else if (ArgType->isFloatTy()) {
        // Logger::logs("proteus") << "RC is Float\n";
        C = ConstantFP::get(ArgType, RC.Value.FloatVal);
      } else if (ArgType->isDoubleTy()) {
        // NOTE: long double on device should correspond to plain double.
        // XXX: CUDA with a long double SILENTLY fails to create a working
        // kernel in AOT compilation, with or without JIT.
        if (RC.Type == RuntimeConstantType::LONG_DOUBLE)
          C = ConstantFP::get(ArgType, RC.Value.LongDoubleVal);
        else
          C = ConstantFP::get(ArgType, RC.Value.DoubleVal);
      } else if (ArgType->isX86_FP80Ty() || ArgType->isPPC_FP128Ty() ||
                 ArgType->isFP128Ty()) {
        C = ConstantFP::get(ArgType, RC.Value.LongDoubleVal);
      } else if (ArgType->isPointerTy()) {
        auto *IntC = ConstantInt::get(Type::getInt64Ty(Ctx), RC.Value.Int64Val);
        C = ConstantExpr::getIntToPtr(IntC, ArgType);
      } else {
        std::string TypeString;
        raw_string_ostream TypeOstream(TypeString);
        ArgType->print(TypeOstream);
        PROTEUS_FATAL_ERROR("JIT Incompatible type in runtime constant: " +
                            TypeOstream.str());
      }

      auto TraceOut = [](Function &F, int ArgNo, Constant *C) {
        SmallString<128> S;
        raw_svector_ostream OS(S);
        OS << "[ArgSpec] Replaced Function " << F.getName() << " ArgNo "
           << ArgNo << " with value " << *C << "\n";

        return S;
      };

      PROTEUS_DBG(Logger::logs("proteus") << TraceOut(F, ArgNo, C));
      if (Config::get().ProteusTraceOutput)
        Logger::trace(TraceOut(F, ArgNo, C));
      Arg->replaceAllUsesWith(C);
    }
  }
};

} // namespace proteus

#endif
