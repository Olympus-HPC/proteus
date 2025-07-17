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

#include <cstdint>
#include <cstring>
#include <optional>

#include "proteus/Error.h"

namespace proteus {

enum RuntimeConstantType : int32_t {
  BEGIN,
  NONE = 0,
  BOOL = 1,
  INT8,
  INT32,
  INT64,
  FLOAT,
  DOUBLE,
  LONG_DOUBLE,
  PTR,
  ARRAY,
  END
};

// This struct holds the information passed from the compiler pass for a runtime
// constant argument to a function.
struct RuntimeConstantArgInfo {
  RuntimeConstantType Type;
  int32_t Pos;

  explicit RuntimeConstantArgInfo(RuntimeConstantType Type, int32_t Pos)
      : Type(Type), Pos(Pos) {}
};

// This struct holds the information from the compiler pass for a runtime
// constant array, that is the number of elements, if a known compile time
// constant, or a runtime constant argument as the number of elements, and the
// element type.
struct RuntimeConstantArrayInfo {
  int32_t NumElts = 0;
  RuntimeConstantType EltType;

  std::optional<RuntimeConstantArgInfo> OptNumEltsRCInfo = std::nullopt;

  explicit RuntimeConstantArrayInfo(int32_t NumElts,
                                    RuntimeConstantType EltType)
      : NumElts(NumElts), EltType(EltType) {}
  explicit RuntimeConstantArrayInfo(RuntimeConstantType EltType,
                                    RuntimeConstantType NumEltsType,
                                    int32_t NumEltsPos)
      : EltType(EltType),
        OptNumEltsRCInfo{RuntimeConstantArgInfo{NumEltsType, NumEltsPos}} {}
};

// This struct holds the information from the compiler pass for a runtime
// constant, be it a scalar or an array. If the runtime constant is an array,
// there is an optional variable to store the runtime constant array info.
struct RuntimeConstantInfo {
  RuntimeConstantArgInfo ArgInfo;
  std::optional<RuntimeConstantArrayInfo> OptArrInfo = std::nullopt;

  explicit RuntimeConstantInfo(RuntimeConstantType Type, int32_t Pos)
      : ArgInfo{Type, Pos} {
    if (Type == RuntimeConstantType::ARRAY)
      PROTEUS_FATAL_ERROR("Missing array info");
  }

  explicit RuntimeConstantInfo(RuntimeConstantType Type, int32_t Pos,
                               int32_t NumElts, RuntimeConstantType EltType)
      : ArgInfo{Type, Pos},
        OptArrInfo{RuntimeConstantArrayInfo{NumElts, EltType}} {
    if (Type != RuntimeConstantType::ARRAY)
      PROTEUS_FATAL_ERROR("Expected array runtime constant but type is " +
                          std::to_string(Type));
  }

  explicit RuntimeConstantInfo(RuntimeConstantType Type, int32_t Pos,
                               RuntimeConstantType EltType,
                               RuntimeConstantType NumEltsType,
                               int32_t NumEltsPos)
      : ArgInfo{Type, Pos},
        OptArrInfo{RuntimeConstantArrayInfo{EltType, NumEltsType, NumEltsPos}} {
    if (Type != RuntimeConstantType::ARRAY)
      PROTEUS_FATAL_ERROR("Expected array runtime constant but type is " +
                          std::to_string(Type));
  }

  bool operator==(const RuntimeConstantInfo &O) const {
    return ((ArgInfo.Type == O.ArgInfo.Type) && (ArgInfo.Pos == O.ArgInfo.Pos));
  }
  bool operator!=(const RuntimeConstantInfo &O) const { return !(*this == O); }

  // Compare by Pos.
  bool operator<(const RuntimeConstantInfo &O) const {
    return ArgInfo.Pos < O.ArgInfo.Pos;
  }
};

// This struct holds the concrete array information, the number of elements and
// element type, for the runtime library, constructed using the runtime constant
// info provided by the compiler pass.
struct ArrayInfo {
  int32_t NumElts;
  RuntimeConstantType EltType;
};

// This union stores all possible runtime constant values.
// TODO: Try std::variant as better type-checked interface, check performance
// implications.
union RuntimeConstantValue {
  bool BoolVal;
  int8_t Int8Val;
  int32_t Int32Val;
  int64_t Int64Val;
  float FloatVal;
  double DoubleVal;
  long double LongDoubleVal;
  void *PtrVal;
};

// This struct holds all information used by the runtime library for a runtime
// constant, be it a scalar or an array, for specialization.
struct RuntimeConstant {
  RuntimeConstantValue Value;
  RuntimeConstantType Type;
  int32_t Pos;
  int32_t Slot{-1};

  std::optional<ArrayInfo> OptArrInfo = std::nullopt;

  explicit RuntimeConstant(RuntimeConstantType Type, int32_t Pos)
      : Type(Type), Pos(Pos) {
    std::memset(&Value, 0, sizeof(RuntimeConstantValue));
  }

  explicit RuntimeConstant(RuntimeConstantType Type, int32_t Pos,
                           int32_t NumElts, RuntimeConstantType EltType)
      : Type(Type), Pos(Pos), OptArrInfo{ArrayInfo{NumElts, EltType}} {
    std::memset(&Value, 0, sizeof(RuntimeConstantValue));
  }

  explicit RuntimeConstant() {
    std::memset(&Value, 0, sizeof(RuntimeConstantValue));
  }

  RuntimeConstant(const RuntimeConstant &) = default;
  RuntimeConstant(RuntimeConstant &&) = default;
  RuntimeConstant &operator=(const RuntimeConstant &) = default;
  RuntimeConstant &operator=(RuntimeConstant &&) = default;
};

} // namespace proteus

#endif
