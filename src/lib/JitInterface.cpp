//===-- JitInterface.cpp -- Proteus user-facing init/finalize/enable/disable
//--===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "proteus/JitInterface.h"
#include "proteus/Init.h"

// NOLINTBEGIN(readability-identifier-naming)
extern "C" void __jit_init_host();
extern "C" void __jit_init_device();
extern "C" void __jit_finalize_host();
extern "C" void __jit_finalize_device();
extern "C" void __jit_enable_host();
extern "C" void __jit_enable_device();
extern "C" void __jit_disable_host();
extern "C" void __jit_disable_device();
// NOLINTEND(readability-identifier-naming)

namespace proteus {

void init() {
  proteusIsInitialized() = true;
  __jit_init_host();
#if PROTEUS_ENABLE_HIP || PROTEUS_ENABLE_CUDA
  __jit_init_device();
#endif
}

void finalize() {
  __jit_finalize_host();
#if PROTEUS_ENABLE_HIP || PROTEUS_ENABLE_CUDA
  __jit_finalize_device();
#endif
}

void enable() {
  __jit_enable_host();
#if PROTEUS_ENABLE_HIP || PROTEUS_ENABLE_CUDA
  __jit_enable_device();
#endif
}

void disable() {
  __jit_disable_host();
#if PROTEUS_ENABLE_HIP || PROTEUS_ENABLE_CUDA
  __jit_disable_device();
#endif
}

} // namespace proteus
