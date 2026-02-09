//===-- TransformSharedArray.h -- Shared array with specialized size--===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_TRANSFORM_SHARED_ARRAY_H
#define PROTEUS_TRANSFORM_SHARED_ARRAY_H

#include "proteus/Debug.h"
#include "proteus/Logger.h"
#include "proteus/Utils.h"

#include <llvm/Analysis/ConstantFolding.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/Debug.h>

namespace proteus {

using namespace llvm;

class TransformSharedArray {
public:
  static void transform(Module &M) {
    for (auto &Func : M.functions()) {
      std::string DemangledName = llvm::demangle(Func.getName().str());
      StringRef StrRef{DemangledName};
      if (StrRef.contains("proteus::shared_array")) {
        // Use a while loop to delete while iterating.
        while (!Func.user_empty()) {
          User *Usr = *Func.user_begin();
          if (!isa<CallBase>(Usr))
            reportFatalError("Expected call user");

          CallBase *CB = cast<CallBase>(Usr);
          assert(CB->arg_size() == 2 && "Expected 2 arguments: N and sizeof");
          int64_t N;
          int64_t Sizeof;
          if (!getConstantValue(CB->getArgOperand(0), N, M.getDataLayout()))
            reportFatalError("Expected constant N argument");
          if (!getConstantValue(CB->getArgOperand(1), Sizeof,
                                M.getDataLayout()))
            reportFatalError("Expected constant Sizeof argument");

          ArrayType *AType =
              ArrayType::get(Type::getInt8Ty(M.getContext()), N * Sizeof);
          constexpr unsigned SharedMemAddrSpace = 3;
          GlobalVariable *SharedMemGV = new GlobalVariable(
              M, AType, false, GlobalValue::InternalLinkage,
              UndefValue::get(AType), ".proteus.shared", nullptr,
              llvm::GlobalValue::NotThreadLocal, SharedMemAddrSpace, false);
          // Using 16-byte alignment based on AOT code generation.
          // TODO: Create or find an API to query the proper ABI alignment.
          SharedMemGV->setAlignment(Align{16});

          auto TraceOut = [](StringRef DemangledName,
                             GlobalVariable *SharedMemGV) {
            SmallString<128> S;
            raw_svector_ostream OS(S);
            OS << "[SharedArray] " << "Replace CB " << DemangledName << " with "
               << *SharedMemGV << "\n";

            return S;
          };

          PROTEUS_DBG(Logger::logs("proteus")
                      << TraceOut(DemangledName, SharedMemGV));
          if (Config::get().ProteusTraceOutput >= 1)
            Logger::trace(TraceOut(DemangledName, SharedMemGV));

          CB->replaceAllUsesWith(ConstantExpr::getAddrSpaceCast(
              SharedMemGV, CB->getFunctionType()->getReturnType()));
          CB->eraseFromParent();
        }

        if (Config::get().ProteusDebugOutput) {
          if (verifyModule(M, &errs()))
            reportFatalError("Broken module found, JIT compilation aborted!");
        }
      }
    }
  }

private:
  static bool getConstantValue(Value *V, int64_t &Result,
                               const DataLayout &DL) {
    // Directly access the value of a constant.
    if (ConstantInt *CI = dyn_cast<ConstantInt>(V)) {
      Result = CI->getSExtValue();
      return true;
    }

    // Fold an instruction.
    if (Instruction *I = dyn_cast<Instruction>(V)) {
      if (Value *FoldedV = ConstantFoldInstruction(I, DL, nullptr)) {
        if (ConstantInt *CI = dyn_cast<ConstantInt>(FoldedV)) {
          Result = CI->getSExtValue();
          return true;
        }
      }
    }

    return false;
  }
};

} // namespace proteus

#endif
