//===-- JitInterface.cpp -- implementation of JitInterface functions --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "proteus/JitInterface.hpp"

namespace proteus {

void enable() {
  __jit_enable_host();
  if constexpr (PROTEUS_ENABLE_CUDA || PROTEUS_ENABLE_HIP) {
  __jit_enable_device();
  }
}

void disable() {
  __jit_disable_host();
  if constexpr (PROTEUS_ENABLE_CUDA || PROTEUS_ENABLE_HIP) {
  __jit_disable_device();
  }
}

void init() {
  __jit_init_host();
  if constexpr (PROTEUS_ENABLE_CUDA || PROTEUS_ENABLE_HIP) {
  __jit_init_device();
  }
}

void finalize() {
  __jit_finalize_host();
  if constexpr (PROTEUS_ENABLE_CUDA || PROTEUS_ENABLE_HIP) {
  __jit_finalize_device();
  }
}

} // namespace proteus
