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
#include "proteus/RuntimeConstantHelpers.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Metadata.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace proteus {

using namespace llvm;

struct AutoCaptureInfo {
  int32_t SlotIndex;
  int32_t ByteOffset;
  RuntimeConstantType Type;
};

inline std::vector<AutoCaptureInfo>
parseAutoReadOnlyCaptureMetadata(const llvm::Function &F) {
  std::vector<AutoCaptureInfo> Captures;
  const MDNode *Root = F.getMetadata("proteus.auto_readonly_captures");
  if (!Root)
    return Captures;

  auto ParseI32 = [](const Metadata *M) -> std::optional<int32_t> {
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
    if (!RuntimeConstantHelpers::isSupportedScalarType(RCType))
      continue;

    Captures.push_back(AutoCaptureInfo{*SlotIndex, *ByteOffset, RCType});
  }

  return Captures;
}

inline std::vector<RuntimeConstant>
extractAutoReadOnlyCapturesFromMetadata(const llvm::Function &F,
                                        const void *ClosureBaseBytes) {
  std::vector<RuntimeConstant> Result;
  if (!ClosureBaseBytes)
    return Result;

  std::vector<AutoCaptureInfo> Captures = parseAutoReadOnlyCaptureMetadata(F);
  if (Captures.empty())
    return Result;

  const char *ClosureBytes = static_cast<const char *>(ClosureBaseBytes);
  Result.reserve(Captures.size());
  for (const auto &Capture : Captures) {
    RuntimeConstant RC{RuntimeConstantType::NONE, Capture.SlotIndex};
    const void *ValuePtr = ClosureBytes + Capture.ByteOffset;
    if (!RuntimeConstantHelpers::tryReadScalar(ValuePtr, Capture.Type,
                                               Capture.SlotIndex, RC))
      continue;
    Result.push_back(RC);
  }

  return Result;
}

} // namespace proteus

#endif
