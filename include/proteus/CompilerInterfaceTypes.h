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

#include <llvm/IR/DerivedTypes.h>
#include <llvm/Support/raw_ostream.h>

#include "proteus/Error.h"

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

inline RuntimeConstantTypes convertTypeToRuntimeConstantType(llvm::Type *Ty) {
  if (Ty->isIntegerTy(1))
    return RuntimeConstantTypes::BOOL;
  if (Ty->isIntegerTy(8))
    return RuntimeConstantTypes::INT8;
  if (Ty->isIntegerTy(32))
    return RuntimeConstantTypes::INT32;
  if (Ty->isIntegerTy(64))
    return RuntimeConstantTypes::INT64;
  if (Ty->isFloatTy())
    return RuntimeConstantTypes::FLOAT;
  if (Ty->isDoubleTy())
    return RuntimeConstantTypes::DOUBLE;
  if (Ty->isFP128Ty() || Ty->isPPC_FP128Ty() || Ty->isX86_FP80Ty())
    return RuntimeConstantTypes::LONG_DOUBLE;
  if (Ty->isPointerTy())
    return RuntimeConstantTypes::PTR;

  std::string TypeString;
  llvm::raw_string_ostream TypeOstream(TypeString);
  Ty->print(TypeOstream);
  PROTEUS_FATAL_ERROR("Unknown Type " + TypeOstream.str());
}

} // namespace proteus

#endif
