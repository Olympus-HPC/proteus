//===-- AutoReadOnlyCaptures.h -- Auto read-only capture metadata utils --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_AUTOREADONLYCAPTURES_H
#define PROTEUS_AUTOREADONLYCAPTURES_H

#include "proteus/CompilerInterfaceTypes.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>

namespace proteus {

using namespace llvm;

struct AutoReadOnlyCaptureMetadataEntry {
  int32_t SlotIndex;
  int32_t ByteOffset;
  RuntimeConstantType RCType;
};

inline bool isSupportedAutoReadOnlyRCType(RuntimeConstantType RCType) {
  switch (RCType) {
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

/// Merge auto-detected captures with explicit captures (explicit takes
/// precedence).
inline void mergeCaptures(llvm::SmallVectorImpl<RuntimeConstant> &Explicit,
                          const llvm::SmallVectorImpl<RuntimeConstant> &Auto) {
  llvm::SmallSet<int32_t, 16> ExplicitSlots;
  for (const auto &RC : Explicit)
    ExplicitSlots.insert(RC.Pos);

  for (const auto &RC : Auto) {
    if (!ExplicitSlots.contains(RC.Pos))
      Explicit.push_back(RC);
  }
}

inline RuntimeConstant
readValueFromMemory(const void *Ptr,
                    const AutoReadOnlyCaptureMetadataEntry &Capture) {
  RuntimeConstant RC(RuntimeConstantType::NONE, Capture.SlotIndex,
                     Capture.ByteOffset);
  RuntimeConstantType RCType = Capture.RCType;

  if (RCType == RuntimeConstantType::BOOL) {
    RC.Type = RuntimeConstantType::BOOL;
    RC.Value.BoolVal = *static_cast<const bool *>(Ptr);
  } else if (RCType == RuntimeConstantType::INT8) {
    RC.Type = RuntimeConstantType::INT8;
    RC.Value.Int8Val = *static_cast<const int8_t *>(Ptr);
  } else if (RCType == RuntimeConstantType::INT32) {
    RC.Type = RuntimeConstantType::INT32;
    RC.Value.Int32Val = *static_cast<const int32_t *>(Ptr);
  } else if (RCType == RuntimeConstantType::INT64) {
    RC.Type = RuntimeConstantType::INT64;
    RC.Value.Int64Val = *static_cast<const int64_t *>(Ptr);
  } else if (RCType == RuntimeConstantType::FLOAT) {
    RC.Type = RuntimeConstantType::FLOAT;
    RC.Value.FloatVal = *static_cast<const float *>(Ptr);
  } else if (RCType == RuntimeConstantType::DOUBLE) {
    RC.Type = RuntimeConstantType::DOUBLE;
    RC.Value.DoubleVal = *static_cast<const double *>(Ptr);
  }

  return RC;
}

inline llvm::SmallVector<AutoReadOnlyCaptureMetadataEntry>
parseAutoReadOnlyCapturesMetadata(Function &F) {
  llvm::SmallVector<AutoReadOnlyCaptureMetadataEntry> Captures;
  MDNode *Root = F.getMetadata("proteus.auto_readonly_captures");
  if (!Root)
    return Captures;

  auto ParseI32 = [](Metadata *M) -> std::optional<int32_t> {
    auto *CAM = dyn_cast<ConstantAsMetadata>(M);
    if (!CAM)
      return std::nullopt;
    auto *CI = dyn_cast<ConstantInt>(CAM->getValue());
    if (!CI)
      return std::nullopt;
    return static_cast<int32_t>(CI->getSExtValue());
  };

  for (unsigned I = 0; I < Root->getNumOperands(); ++I) {
    auto *EntryNode = dyn_cast_or_null<MDNode>(Root->getOperand(I));
    if (!EntryNode || EntryNode->getNumOperands() != 3)
      continue;

    auto SlotIndex = ParseI32(EntryNode->getOperand(0));
    auto ByteOffset = ParseI32(EntryNode->getOperand(1));
    auto RCTypeInt = ParseI32(EntryNode->getOperand(2));
    if (!SlotIndex || !ByteOffset || !RCTypeInt)
      continue;
    if (*SlotIndex < 0 || *ByteOffset < 0)
      continue;

    RuntimeConstantType RCType = static_cast<RuntimeConstantType>(*RCTypeInt);
    if (!isSupportedAutoReadOnlyRCType(RCType))
      continue;

    Captures.push_back(
        AutoReadOnlyCaptureMetadataEntry{*SlotIndex, *ByteOffset, RCType});
  }

  return Captures;
}

inline llvm::SmallVector<RuntimeConstant>
extractAutoDetectedCapturesFromMetadata(
    const void *LambdaClosure,
    const llvm::SmallVector<AutoReadOnlyCaptureMetadataEntry>
        &DetectedCaptures) {
  llvm::SmallVector<RuntimeConstant> Result;
  if (!LambdaClosure || DetectedCaptures.empty())
    return Result;

  const char *ClosureBytes = static_cast<const char *>(LambdaClosure);
  for (const auto &Cap : DetectedCaptures) {
    RuntimeConstant RC =
        readValueFromMemory(ClosureBytes + Cap.ByteOffset, Cap);
    if (RC.Type == RuntimeConstantType::NONE)
      continue;
    Result.push_back(RC);
  }

  return Result;
}

inline SmallString<128> traceOutAuto(int Slot, const RuntimeConstant &RC) {
  SmallString<128> S;
  raw_svector_ostream OS(S);
  OS << "[LambdaSpec][Auto] Replacing slot " << Slot << " with ";

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
    OS << "float " << format("%g", RC.Value.FloatVal);
    break;
  case RuntimeConstantType::DOUBLE:
    OS << "double " << format("%g", RC.Value.DoubleVal);
    break;
  default:
    OS << "<unsupported type>";
    break;
  }

  OS << "\n";
  return S;
}

} // namespace proteus

#endif
