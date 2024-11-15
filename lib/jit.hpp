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
#include "JitEngineHost.hpp"

namespace proteus {

template<typename T>
T jit_variable(T v, int pos=-1) {
  JitEngineHost &Jit = JitEngineHost::instance();

  RuntimeConstant RC;
  std::memcpy(&RC, &v, sizeof(T));
  RC.Slot = pos;
  Jit.pushJitVariable(RC);

  return v;
}

// template<typename T> struct jit_variable {
//   const T value;
//   jit_variable(const T v) : value(v) {}
//   operator T() { return value; }
//   operator T() const { return value; }
// };

}