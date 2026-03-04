//===-- LambdaCaptureSpecialization.h -- Lambda capture merge helpers --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_LAMBDACAPTURESPECIALIZATION_H
#define PROTEUS_LAMBDACAPTURESPECIALIZATION_H

#include "proteus/AutoReadOnlyCaptures.h"
#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/Config.h"
#include "proteus/Logger.h"
#include "proteus/RuntimeConstantHelpers.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Function.h>

#include <string>

namespace proteus {

inline void mergeExplicitAndAuto(llvm::SmallVectorImpl<RuntimeConstant> &Out,
                                 llvm::ArrayRef<RuntimeConstant> Explicit,
                                 const llvm::Function *LambdaFn,
                                 const void *ClosureBytes) {
  Out.clear();
  Out.insert(Out.end(), Explicit.begin(), Explicit.end());

  if (!LambdaFn || !ClosureBytes)
    return;

  auto AutoCaptures =
      extractAutoReadOnlyCapturesFromMetadata(*LambdaFn, ClosureBytes);
  for (const auto &RC : AutoCaptures) {
    bool WasExplicit = false;
    for (const auto &ExplicitRC : Explicit) {
      if (ExplicitRC.Pos == RC.Pos) {
        WasExplicit = true;
        break;
      }
    }

    if (WasExplicit)
      continue;

    Out.push_back(RC);

    if (Config::get().ProteusTraceOutput >= 1) {
      Logger::trace("[LambdaSpec][Auto] Replacing slot " +
                    std::to_string(RC.Pos) + " with " +
                    RuntimeConstantHelpers::toString(RC) + "\n");
    }
  }
}

} // namespace proteus

#endif
