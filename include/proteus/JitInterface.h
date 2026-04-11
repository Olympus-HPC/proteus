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
#include "proteus/Init.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>

extern "C" void __jit_register_variable(proteus::RuntimeConstant RC,
                                        const char *AssociatedLambda);

extern "C" __attribute__((used)) void __jit_register_lambda_runtime_constant(
    int32_t Type, int32_t Pos, int32_t Offset, const void *ValuePtr, uint64_t functor_id);

extern "C" void __jit_register_lambda(const char *Symbol);
extern "C" void __jit_take_address(void const *) noexcept;
extern "C" void __jit_var(void const *) noexcept;

namespace proteus {

#if defined(__CUDACC__) || defined(__HIP__)
#define PROTEUS_HOST_DEVICE __host__ __device__
#else
#define PROTEUS_HOST_DEVICE
#endif

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
  } else {
    return RuntimeConstantType::NONE;
  }
}

namespace detail {
  constexpr std::uint64_t fnv1a64(const char* s) {
    std::uint64_t h = 14695981039346656037ull;
    for (; *s; ++s) { h ^= (unsigned char)(*s); h *= 1099511628211ull; }
    return h;
  }

  template <class Lambda, std::uint32_t Ctr>
  constexpr std::uint64_t functor_id() {
    return fnv1a64(__PRETTY_FUNCTION__); // includes Lambda + Ctr in the text
  }
  } // namespace proteus::detail

template <typename T>
static __attribute__((noinline))  T
jit_variable(T V) noexcept {return V; }

template <uint64_t FunctorID, typename Lambda>
struct LambdaFunctorWrapper {
  using LambdaType = Lambda;
  static constexpr std::uint64_t functor_id = FunctorID;
  LambdaType lambda;
  // breadcrumb tying device and host modules together.
  PROTEUS_HOST_DEVICE __attribute__((noinline)) void register_functor(){}

  template <typename... Args>
  PROTEUS_HOST_DEVICE __attribute__((annotate("jit")))
  __attribute__((annotate("proteus.wrapper_call", functor_id))) decltype(auto)
  operator()(Args &&...args) noexcept(
      noexcept(lambda(std::forward<Args>(args)...))) {
    register_functor();
    return lambda(std::forward<Args>(args)...);
  }
};

template <std::uint64_t FunctorId, typename L>
PROTEUS_HOST_DEVICE inline auto
tag_functor(L &&lambda) {
  return LambdaFunctorWrapper<FunctorId, std::decay_t<L>>{std::forward<L>(lambda)};
}

template <uint64_t ID, typename T>
[[nodiscard]] static __attribute__((noinline)) __attribute__((noinline)) __attribute__((annotate("proteus.register_call", ID))) auto
__register_lambda_impl(T &&t, const char *Symbol = "") noexcept {
  assert(Symbol && "Expected non-null Symbol");
  __jit_register_lambda(Symbol);
  // Force LLVM to generate an AllocaInst of the underlying Clang--generated
  // anonymous class for T.  We remove this after recording the demangled
  // lambda name.
  using LambdaType = std::decay_t<T>;
  LambdaType local = t;
  __jit_take_address(&local);
  auto result = tag_functor<ID>(std::forward<T>(t));
  result.register_functor();
  return result;
}


#define PROTEUS_REGISTER_LAMBDA(lam) PROTEUS_REGISTER_LAMBDA_IMPL(lam, __COUNTER__)
#define PROTEUS_REGISTER_LAMBDA_IMPL(lam, ctr) \
  ::proteus::__register_lambda_impl< \
    ::proteus::detail::functor_id<std::decay_t<decltype(lam)>, (ctr)>() \
  >((lam))

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

} // namespace proteus

#endif

// NOLINTEND(readability-identifier-naming)
