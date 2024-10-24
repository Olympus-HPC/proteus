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

#include "CompilerInterfaceTypes.h"
#include "Utils.h"

namespace proteus {

using namespace llvm;

class TransformArgumentSpecialization {
public:
  static void transform(Module &M, Function &F, ArrayRef<RuntimeConstant> RC) {
    auto &Ctx = M.getContext();
    MDNode *Node = F.getMetadata("jit_arg_nos");
    DBG(dbgs() << "Metadata jit for F " << F.getName() << " = " << *Node
               << "\n");

    // Replace argument uses with runtime constants.
    for (int I = 0; I < Node->getNumOperands(); ++I) {
      ConstantAsMetadata *CAM = cast<ConstantAsMetadata>(Node->getOperand(I));
      ConstantInt *ConstInt = cast<ConstantInt>(CAM->getValue());
      int ArgNo = ConstInt->getZExtValue();
      Value *Arg = F.getArg(ArgNo);
      Type *ArgType = Arg->getType();
      Constant *C = nullptr;
      if (ArgType->isIntegerTy(1)) {
        C = ConstantInt::get(ArgType, RC[I].BoolVal);
      } else if (ArgType->isIntegerTy(8)) {
        C = ConstantInt::get(ArgType, RC[I].Int8Val);
      } else if (ArgType->isIntegerTy(32)) {
        // dbgs() << "RC is Int32\n";
        C = ConstantInt::get(ArgType, RC[I].Int32Val);
      } else if (ArgType->isIntegerTy(64)) {
        // dbgs() << "RC is Int64\n";
        C = ConstantInt::get(ArgType, RC[I].Int64Val);
      } else if (ArgType->isFloatTy()) {
        // dbgs() << "RC is Float\n";
        C = ConstantFP::get(ArgType, RC[I].FloatVal);
      }
      // NOTE: long double on device should correspond to plain double.
      // XXX: CUDA with a long double SILENTLY fails to create a working
      // kernel in AOT compilation, with or without JIT.
      else if (ArgType->isDoubleTy()) {
        // dbgs() << "RC is Double\n";
        C = ConstantFP::get(ArgType, RC[I].DoubleVal);
      } else if (ArgType->isX86_FP80Ty() || ArgType->isPPC_FP128Ty() ||
                 ArgType->isFP128Ty()) {
        C = ConstantFP::get(ArgType, RC[I].LongDoubleVal);
      } else if (ArgType->isPointerTy()) {
        auto *IntC = ConstantInt::get(Type::getInt64Ty(Ctx), RC[I].Int64Val);
        C = ConstantExpr::getIntToPtr(IntC, ArgType);
      } else {
        std::string TypeString;
        llvm::raw_string_ostream TypeOstream(TypeString);
        ArgType->print(TypeOstream);
        FATAL_ERROR("JIT Incompatible type in runtime constant: " +
                    TypeOstream.str());
      }

      Arg->replaceAllUsesWith(C);
    }
  }
};

} // namespace proteus

#endif
