//===-- jit.hpp -- user interface to Proteus JIT library --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

// NOLINTBEGIN(readability-identifier-naming)

#ifndef PROTEUS_JIT_INTERFACE_HPP
#define PROTEUS_JIT_INTERFACE_HPP

#include "proteus/CompilerInterfaceTypes.h"

#include <cassert>
#include <cstring>
#include <utility>

extern "C" void __jit_push_variable(proteus::RuntimeConstant RC);
extern "C" void __jit_register_lambda(const char *Symbol);
extern "C" void __jit_init_host();
extern "C" void __jit_init_device();
extern "C" void __jit_finalize_host();
extern "C" void __jit_finalize_device();
extern "C" void __jit_enable_host();
extern "C" void __jit_enable_device();
extern "C" void __jit_disable_host();
extern "C" void __jit_disable_device();

namespace proteus {

template <typename T> __attribute__((noinline)) void jit_arg(T V) noexcept;
#if defined(__CUDACC__) || defined(__HIP__)
template <typename T>
__attribute__((noinline)) __device__ void jit_arg(T V) noexcept;
#endif

template <typename T>
static __attribute__((noinline)) T jit_variable(T v, int pos = -1) {
  RuntimeConstant RC;
  std::memcpy(&RC, &v, sizeof(T));
  RC.Slot = pos;
  __jit_push_variable(RC);

  return v;
}

template <typename T>
static __attribute__((noinline)) T &&register_lambda(T &&t,
                                                     const char *Symbol = "") {
  assert(Symbol && "Expected non-null Symbol");
  __jit_register_lambda(Symbol);
  return std::forward<T>(t);
}

#if defined(__CUDACC__) || defined(__HIP__)
template <typename T>
__device__ __attribute__((noinline)) T *
shared_array([[maybe_unused]] size_t N,
             [[maybe_unused]] size_t ElemSize = sizeof(T)) {
  __builtin_trap();
}
#endif

inline void enable() {
  __jit_enable_host();
  __jit_enable_device();
}

inline void disable() {
  __jit_disable_host();
  __jit_disable_device();
}

inline void init() {
  __jit_init_host();
#if PROTEUS_ENABLE_HIP || PROTEUS_ENABLE_CUDA
  __jit_init_device();
#endif
}

inline void finalize() {
  __jit_finalize_host();
#if PROTEUS_ENABLE_HIP || PROTEUS_ENABLE_CUDA
  __jit_finalize_device();
#endif
}

} // namespace proteus

#endif

// NOLINTBEGIN(readability-identifier-naming)
