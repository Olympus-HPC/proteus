//===-- Init.cpp -- Proteus initialization/finalization --------------===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------===//

#include "proteus/Init.h"

// NOLINTBEGIN(readability-identifier-naming)
extern "C" void __proteus_enable_host();
extern "C" void __proteus_enable_device();
extern "C" void __proteus_disable_host();
extern "C" void __proteus_disable_device();
// NOLINTEND(readability-identifier-naming)

namespace proteus {

void init() {}
void finalize() {}

void enable() {
  __proteus_enable_host();
#if PROTEUS_ENABLE_HIP || PROTEUS_ENABLE_CUDA
  __proteus_enable_device();
#endif
}

void disable() {
  __proteus_disable_host();
#if PROTEUS_ENABLE_HIP || PROTEUS_ENABLE_CUDA
  __proteus_disable_device();
#endif
}

} // namespace proteus
