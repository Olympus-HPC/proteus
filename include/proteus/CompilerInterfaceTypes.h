//===-- CompilerInterfaceTypes.cpp -- JIT compiler interface types --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_COMPILERINTERFACETYPES_H
#define PROTEUS_COMPILERINTERFACETYPES_H

#include <cstring>
#include <stdint.h>

namespace proteus {

enum RuntimeConstantTypes : int32_t {
  BOOL = 1,
  INT8,
  INT32,
  INT64,
  FLOAT,
  DOUBLE,
  LONG_DOUBLE,
  PTR
};

struct RuntimeConstant {
  RuntimeConstant() { std::memset(&Value, 0, sizeof(RuntimeConstantType)); }
  using RuntimeConstantType = union {
    bool BoolVal;
    int8_t Int8Val;
    int32_t Int32Val;
    int64_t Int64Val;
    float FloatVal;
    double DoubleVal;
    long double LongDoubleVal;
    // TODO: This allows pointer as runtime constant values. How useful is that?
    void *PtrVal;
  };
  RuntimeConstantType Value;
  int32_t Slot{-1};
};

} // namespace proteus

#endif
