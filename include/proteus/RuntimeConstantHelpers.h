//===-- RuntimeConstantHelpers.h -- RuntimeConstant utility helpers --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_RUNTIME_CONSTANT_HELPERS_H
#define PROTEUS_RUNTIME_CONSTANT_HELPERS_H

#include "proteus/CompilerInterfaceTypes.h"

#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

#include <cstdint>
#include <cstring>
#include <string>

namespace proteus {
namespace RuntimeConstantHelpers {

inline bool isSupportedScalarType(RuntimeConstantType Type) {
  switch (Type) {
  case RuntimeConstantType::BOOL:
  case RuntimeConstantType::INT8:
  case RuntimeConstantType::INT32:
  case RuntimeConstantType::INT64:
  case RuntimeConstantType::FLOAT:
  case RuntimeConstantType::DOUBLE:
    return true;
  default:
    return false;
  }
}

template <typename T> inline T readUnalignedValue(const void *Ptr) {
  T Value{};
  std::memcpy(&Value, Ptr, sizeof(T));
  return Value;
}

inline bool tryReadScalar(const void *Ptr, RuntimeConstantType Type,
                          int32_t Pos, RuntimeConstant &Out) {
  Out = RuntimeConstant(RuntimeConstantType::NONE, Pos);

  switch (Type) {
  case RuntimeConstantType::BOOL:
    Out.Type = RuntimeConstantType::BOOL;
    Out.Value.BoolVal = readUnalignedValue<bool>(Ptr);
    return true;
  case RuntimeConstantType::INT8:
    Out.Type = RuntimeConstantType::INT8;
    Out.Value.Int8Val = readUnalignedValue<int8_t>(Ptr);
    return true;
  case RuntimeConstantType::INT32:
    Out.Type = RuntimeConstantType::INT32;
    Out.Value.Int32Val = readUnalignedValue<int32_t>(Ptr);
    return true;
  case RuntimeConstantType::INT64:
    Out.Type = RuntimeConstantType::INT64;
    Out.Value.Int64Val = readUnalignedValue<int64_t>(Ptr);
    return true;
  case RuntimeConstantType::FLOAT:
    Out.Type = RuntimeConstantType::FLOAT;
    Out.Value.FloatVal = readUnalignedValue<float>(Ptr);
    return true;
  case RuntimeConstantType::DOUBLE:
    Out.Type = RuntimeConstantType::DOUBLE;
    Out.Value.DoubleVal = readUnalignedValue<double>(Ptr);
    return true;
  default:
    return false;
  }
}

inline std::string toString(const RuntimeConstant &RC) {
  std::string S;
  llvm::raw_string_ostream OS(S);
  switch (RC.Type) {
  case RuntimeConstantType::BOOL:
    OS << "i1 " << (RC.Value.BoolVal ? "1" : "0");
    break;
  case RuntimeConstantType::INT8:
    OS << "i8 " << static_cast<int>(RC.Value.Int8Val);
    break;
  case RuntimeConstantType::INT32:
    OS << "i32 " << RC.Value.Int32Val;
    break;
  case RuntimeConstantType::INT64:
    OS << "i64 " << RC.Value.Int64Val;
    break;
  case RuntimeConstantType::FLOAT:
    OS << "float " << llvm::format("%g", RC.Value.FloatVal);
    break;
  case RuntimeConstantType::DOUBLE:
    OS << "double " << llvm::format("%g", RC.Value.DoubleVal);
    break;
  default:
    OS << "<unsupported type>";
    break;
  }
  return OS.str();
}

} // namespace RuntimeConstantHelpers
} // namespace proteus

#endif