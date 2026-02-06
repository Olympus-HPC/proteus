//===-- Init.h -- Proteus initialization state --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_INIT_H
#define PROTEUS_INIT_H

#include "proteus/Error.h"

namespace proteus {

inline bool &proteusIsInitialized() {
  static bool Initialized = false;
  return Initialized;
}

inline void ensureProteusInitialized() {
  if (!isInitialized())
    reportFatalError(
        "proteus not initialized. Call proteus::init() before using JIT "
        "compilation.");
}

} // namespace proteus

#endif
