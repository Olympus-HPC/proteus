//===-- AutoReadOnlyCaptures.h -- Auto-detect read-only captures --===//
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

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/DataLayout.h"

namespace proteus {

using namespace llvm;

/// Information about a detected lambda capture
struct CaptureInfo {
  int32_t Offset;              // Byte offset within lambda closure
  int32_t SlotIndex;           // GEP slot index for struct access (0-based)
  llvm::Type *CaptureType;     // LLVM type of the capture
  bool IsReadOnly;             // Whether capture is read-only
};

/// Check if a type is a supported scalar type for auto-detection
inline bool isSupportedScalarType(llvm::Type *Ty) {
  if (Ty->isIntegerTy(1) || Ty->isIntegerTy(8) ||
      Ty->isIntegerTy(32) || Ty->isIntegerTy(64))
    return true;
  if (Ty->isFloatTy() || Ty->isDoubleTy())
    return true;
  return false;
}

/// Conservative escape analysis: returns true if pointer escapes
inline bool pointerEscapes(llvm::Value *V) {
  for (auto *User : V->users()) {
    if (isa<StoreInst>(User))
      return true;  // Pointer stored somewhere
    if (isa<CallInst>(User) || isa<InvokeInst>(User))
      return true;  // Pointer passed to function
    if (auto *GEP = dyn_cast<GetElementPtrInst>(User)) {
      if (pointerEscapes(GEP))  // Recurse for derived pointers
        return true;
    }
    // LoadInst is fine - just reading the value
  }
  return false;
}

/// Merge auto-detected captures with explicit captures (explicit takes precedence)
inline void mergeCaptures(llvm::SmallVectorImpl<RuntimeConstant> &Explicit,
                          const llvm::SmallVectorImpl<RuntimeConstant> &Auto) {
  // Build set of slots already covered by explicit captures
  llvm::SmallSet<int32_t, 16> ExplicitSlots;
  for (const auto &RC : Explicit)
    ExplicitSlots.insert(RC.Pos);

  // Add auto-detected captures that don't conflict with explicit ones
  for (const auto &RC : Auto) {
    if (!ExplicitSlots.contains(RC.Pos))
      Explicit.push_back(RC);
  }
}

/// Analyze a JIT lambda function to detect read-only captures
inline llvm::SmallVector<CaptureInfo> analyzeReadOnlyCaptures(Function &F) {
  llvm::SmallVector<CaptureInfo> Captures;

  // Get the lambda closure argument (first argument)
  if (F.arg_empty())
    return Captures;

  Argument *ClosureArg = &*F.arg_begin();

  // Track which slots have been seen and whether they're read-only
  llvm::DenseMap<int32_t, CaptureInfo> SlotInfo;

  // Analyze all uses of the closure argument
  for (User *User : ClosureArg->users()) {
    // Case 1: Direct LoadInst (single-value capture at slot 0)
    if (auto *LI = dyn_cast<LoadInst>(User)) {
      Type *LoadType = LI->getType();
      if (!isSupportedScalarType(LoadType))
        continue;

      int32_t SlotIndex = 0;
      if (SlotInfo.find(SlotIndex) == SlotInfo.end()) {
        SlotInfo[SlotIndex] = {0, SlotIndex, LoadType, true};
      }

      // Check if the loaded value is used in a way that makes it not read-only
      if (pointerEscapes(LI)) {
        SlotInfo[SlotIndex].IsReadOnly = false;
      }
      continue;
    }

    // Case 2: GetElementPtrInst (struct field access)
    if (auto *GEP = dyn_cast<GetElementPtrInst>(User)) {
      // For struct access: GEP ptr, 0, fieldIndex
      if (GEP->getNumIndices() >= 2) {
        if (auto *CI = dyn_cast<ConstantInt>(GEP->getOperand(2))) {
          int32_t SlotIndex = CI->getSExtValue();

          // Analyze users of this GEP
          bool IsReadOnly = true;
          Type *CaptureType = nullptr;

          for (User *GEPUser : GEP->users()) {
            // Check for stores to this slot
            if (auto *SI = dyn_cast<StoreInst>(GEPUser)) {
              if (SI->getPointerOperand() == GEP) {
                IsReadOnly = false;
              }
            }
            // Get the capture type from loads
            else if (auto *LI = dyn_cast<LoadInst>(GEPUser)) {
              if (!CaptureType)
                CaptureType = LI->getType();
            }
          }

          // Check if the GEP itself escapes
          if (pointerEscapes(GEP)) {
            IsReadOnly = false;
          }

          // Only add if we found a capture type and it's supported
          if (CaptureType && isSupportedScalarType(CaptureType)) {
            if (SlotInfo.find(SlotIndex) == SlotInfo.end()) {
              // Compute byte offset from DataLayout if available
              int32_t Offset = 0;
              if (auto *STy = dyn_cast<StructType>(GEP->getSourceElementType())) {
                if (const DataLayout *DL = &F.getParent()->getDataLayout()) {
                  const StructLayout *SL = DL->getStructLayout(STy);
                  Offset = SL->getElementOffset(SlotIndex);
                }
              }

              SlotInfo[SlotIndex] = {Offset, SlotIndex, CaptureType, IsReadOnly};
            } else {
              // Update read-only status if we found a store
              if (!IsReadOnly)
                SlotInfo[SlotIndex].IsReadOnly = false;
            }
          }
        }
      }
    }
  }

  // Collect only read-only captures with supported scalar types
  for (const auto &Entry : SlotInfo) {
    const CaptureInfo &Info = Entry.second;
    if (Info.IsReadOnly && isSupportedScalarType(Info.CaptureType)) {
      Captures.push_back(Info);
    }
  }

  return Captures;
}

} // namespace proteus

#endif
