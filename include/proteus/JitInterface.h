//===-- jit.h -- user interface to Proteus JIT library --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

// NOLINTBEGIN(readability-identifier-naming)

#ifndef PROTEUS_JIT_INTERFACE_H
#define PROTEUS_JIT_INTERFACE_H

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
__attribute__((noinline)) void
jit_array(T V, [[maybe_unused]] size_t NumElts,
          [[maybe_unused]]
          typename std::remove_pointer<T>::type Velem = 0) noexcept;
#if defined(__CUDACC__) || defined(__HIP__)
template <typename T>
__attribute__((noinline)) __device__ void
jit_array(T V, [[maybe_unused]] size_t NumElts,
          [[maybe_unused]]
          typename std::remove_pointer<T>::type Velem = 0) noexcept;
#endif

template <typename T>
__attribute__((noinline))
std::enable_if_t<std::is_trivially_copyable_v<std::remove_pointer_t<T>>, void>
jit_object(T *V, size_t Size = sizeof(std::remove_pointer_t<T>)) noexcept;

#if defined(__CUDACC__) || defined(__HIP__)
template <typename T>
__attribute__((noinline)) __device__ std::enable_if_t<
    std::is_trivially_copyable_v<std::remove_pointer_t<T>>, void>
jit_object(T *V, size_t Size = sizeof(T)) noexcept;
#endif

template <typename T>
__attribute__((noinline))
std::enable_if_t<!std::is_pointer_v<T> &&
                     std::is_trivially_copyable_v<std::remove_reference_t<T>>,
                 void>
jit_object(T &V, size_t Size = sizeof(std::remove_reference_t<T>)) noexcept;

#if defined(__CUDACC__) || defined(__HIP__)
template <typename T>
__attribute__((noinline)) __device__ std::enable_if_t<
    !std::is_pointer_v<T> &&
        std::is_trivially_copyable_v<std::remove_reference_t<T>>,
    void>
jit_object(T &V, size_t Size = sizeof(T)) noexcept;
#endif

template <typename T> inline static RuntimeConstantType convertCTypeToRCType() {
  if constexpr (std::is_same_v<T, bool>) {
    return RuntimeConstantType::BOOL;
  } else if constexpr (std::is_integral_v<T> && sizeof(T) == sizeof(int8_t)) {
    return RuntimeConstantType::INT8;
  } else if constexpr (std::is_integral_v<T> && sizeof(T) == sizeof(int32_t)) {
    return RuntimeConstantType::INT32;
  } else if constexpr (std::is_integral_v<T> && sizeof(T) == sizeof(int64_t)) {
    return RuntimeConstantType::INT64;
  } else if constexpr (std::is_same_v<T, float>) {
    return RuntimeConstantType::FLOAT;
  } else if constexpr (std::is_same_v<T, double>) {
    return RuntimeConstantType::DOUBLE;
  } else if constexpr (std::is_same_v<T, long double>) {
    return RuntimeConstantType::LONG_DOUBLE;
  } else if constexpr (std::is_pointer_v<T>) {
    return RuntimeConstantType::PTR;
  } else if constexpr (std::is_enum_v<T>) {
    return RuntimeConstantType::ENUM;
  } else {
    return RuntimeConstantType::NONE;
  }
}

template <typename T>
static __attribute__((noinline)) T jit_variable(T V, int Pos = -1,
                                                int Offset = -1) noexcept {
  RuntimeConstant RC{convertCTypeToRCType<T>(), Pos, Offset};
  std::memcpy(static_cast<void *>(&RC), &V, sizeof(T));
  __jit_push_variable(RC);

  return V;
}

template <typename T>
static __attribute__((noinline)) T &&
register_lambda(T &&t, const char *Symbol = "") noexcept {
  assert(Symbol && "Expected non-null Symbol");
  __jit_register_lambda(Symbol);
  return std::forward<T>(t);
}

#if defined(__CUDACC__) || defined(__HIP__)
// The function needs to be static for RDC compilation to resolve the static
// shared memory fallback.
template <typename T, size_t MAXN, int UniqueID = 0>
static __device__ __attribute__((noinline)) T *
shared_array([[maybe_unused]] size_t N,
             [[maybe_unused]] size_t ElemSize = sizeof(T)) {
  alignas(T) static __shared__ char shmem[sizeof(T) * MAXN];
  return reinterpret_cast<T *>(shmem);
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

// NOLINTEND(readability-identifier-naming)
