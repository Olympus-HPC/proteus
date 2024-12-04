//===-- jit.hpp -- user interface to Proteus JIT library --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//
#include "CompilerInterfaceTypes.h"

#include <cstring>

extern "C" void __jit_push_variable(RuntimeConstant RC);
namespace proteus {

template <typename T> T jit_variable(T v, int pos = -1) {

  RuntimeConstant RC;
  std::memcpy(&RC, &v, sizeof(T));
  RC.Slot = pos;
  __jit_push_variable(RC);

  return v;
}
}
