//===-- jit.hpp -- user interface to Proteus JIT library --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_JIT_INTERFACE_HPP
#define PROTEUS_JIT_INTERFACE_HPP

#include "proteus/CompilerInterfaceTypes.h"

#include <cassert>
#include <cstring>
#include <utility>

extern "C" void __jit_push_variable(proteus::RuntimeConstant RC);
extern "C" void __jit_register_lambda(const char *Symbol);

namespace proteus {

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
} // namespace proteus

#endif
