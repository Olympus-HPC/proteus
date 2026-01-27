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
#include <memory>

namespace proteus {

// We don't have a proper ENUM type because we just explicity use the underlying
// integral type
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
  STATIC_ARRAY,
  VECTOR,
  ARRAY,
  OBJECT,
  END
};

// This struct holds the concrete array information, the number of elements and
// element type, for the runtime library, constructed using the runtime constant
// info provided by the compiler pass.
struct ArrayInfo {
  int32_t NumElts;
  RuntimeConstantType EltType;
  std::shared_ptr<unsigned char[]> Blob;
};

// This struct holds the information (the byte size) for an assumed trivially
// copyable object constructed using the runtime constant info provided by the
// compiler pass.
struct ObjectInfo {
  int32_t Size;
  bool PassByValue;
  std::shared_ptr<unsigned char[]> Blob;
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
  int32_t Offset;

  ArrayInfo ArrInfo{0, RuntimeConstantType::NONE, nullptr};
  ObjectInfo ObjInfo{0, false, nullptr};

  explicit RuntimeConstant(RuntimeConstantType Type, int32_t Pos)
      : Type(Type), Pos(Pos) {
    std::memset(&Value, 0, sizeof(RuntimeConstantValue));
  }

  explicit RuntimeConstant(RuntimeConstantType Type, int32_t Pos,
                           int32_t Offset)
      : Type(Type), Pos(Pos), Offset(Offset) {
    std::memset(&Value, 0, sizeof(RuntimeConstantValue));
  }

  explicit RuntimeConstant(RuntimeConstantType Type, int32_t Pos,
                           int32_t NumElts, RuntimeConstantType EltType)
      : Type(Type), Pos(Pos), ArrInfo{ArrayInfo{NumElts, EltType, nullptr}} {
    std::memset(&Value, 0, sizeof(RuntimeConstantValue));
  }

  explicit RuntimeConstant(RuntimeConstantType Type, int32_t Pos, int32_t Size,
                           bool PassByValue)
      : Type(Type), Pos(Pos), ObjInfo{ObjectInfo{Size, PassByValue, nullptr}} {
    std::memset(&Value, 0, sizeof(RuntimeConstantValue));
  }

  RuntimeConstant(const RuntimeConstant &) = default;
  RuntimeConstant(RuntimeConstant &&) = default;
  RuntimeConstant &operator=(const RuntimeConstant &) = default;
  RuntimeConstant &operator=(RuntimeConstant &&) = default;
};

} // namespace proteus

#endif
