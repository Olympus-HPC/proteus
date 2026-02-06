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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

namespace proteus {

using namespace llvm;

/// Information about a detected lambda capture
struct CaptureInfo {
  int32_t Offset;          // Byte offset within lambda closure
  int32_t SlotIndex;       // GEP slot index for struct access (0-based)
  llvm::Type *CaptureType; // LLVM type of the capture
  bool IsReadOnly;         // Whether capture is read-only
};

/// Check if a type is a supported scalar type for auto-detection
inline bool isSupportedScalarType(llvm::Type *Ty) {
  if (Ty->isIntegerTy(1) || Ty->isIntegerTy(8) || Ty->isIntegerTy(32) ||
      Ty->isIntegerTy(64))
    return true;
  if (Ty->isFloatTy() || Ty->isDoubleTy())
    return true;
  return false;
}

/// Conservative escape analysis: returns true if pointer escapes
inline bool pointerEscapes(llvm::Value *V) {
  for (auto *User : V->users()) {
    if (isa<StoreInst>(User))
      return true; // Pointer stored somewhere
    if (isa<CallInst>(User) || isa<InvokeInst>(User))
      return true; // Pointer passed to function
    if (auto *GEP = dyn_cast<GetElementPtrInst>(User)) {
      if (pointerEscapes(GEP)) // Recurse for derived pointers
        return true;
    }
    // LoadInst is fine - just reading the value
  }
  return false;
}

/// Merge auto-detected captures with explicit captures (explicit takes
/// precedence)
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
      int32_t ByteOffset = -1;
      int32_t SlotIndex = -1;

      // Handle two GEP patterns:
      // 1. Typed struct GEP: getelementptr %struct.type, ptr, 0, fieldIndex
      // 2. Byte-offset GEP: getelementptr i8, ptr, byteOffset
      if (GEP->getNumIndices() >= 2) {
        // Typed struct GEP - extract field index from second index
        if (auto *CI = dyn_cast<ConstantInt>(GEP->getOperand(2))) {
          SlotIndex = CI->getSExtValue();
          // Compute byte offset from DataLayout if available
          if (auto *STy = dyn_cast<StructType>(GEP->getSourceElementType())) {
            if (const DataLayout *DL = &F.getParent()->getDataLayout()) {
              const StructLayout *SL = DL->getStructLayout(STy);
              ByteOffset = SL->getElementOffset(SlotIndex);
            }
          }
        }
      } else if (GEP->getNumIndices() == 1) {
        // Byte-offset GEP - use byte offset directly as both offset and slot
        if (auto *CI = dyn_cast<ConstantInt>(GEP->getOperand(1))) {
          ByteOffset = CI->getSExtValue();
          // For byte-offset GEPs, use the byte offset itself as the slot index
          // to avoid collisions when multiple captures have different offsets
          SlotIndex = ByteOffset;
        }
      }

      if (SlotIndex >= 0) {
        // Analyze users of this GEP
        bool IsReadOnly = true;
        Type *CaptureType = nullptr;

        for (llvm::User *GEPUser : GEP->users()) {
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
            SlotInfo[SlotIndex] = {ByteOffset, SlotIndex, CaptureType,
                                   IsReadOnly};
          } else {
            // Update read-only status if we found a store
            if (!IsReadOnly)
              SlotInfo[SlotIndex].IsReadOnly = false;
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

inline RuntimeConstant readValueFromMemory(const void *Ptr, Type *Ty,
                                           int32_t SlotIndex) {
  RuntimeConstant RC(RuntimeConstantType::NONE, SlotIndex);

  if (Ty->isIntegerTy(1)) {
    RC.Type = RuntimeConstantType::BOOL;
    RC.Value.BoolVal = *static_cast<const bool *>(Ptr);
  } else if (Ty->isIntegerTy(8)) {
    RC.Type = RuntimeConstantType::INT8;
    RC.Value.Int8Val = *static_cast<const int8_t *>(Ptr);
  } else if (Ty->isIntegerTy(32)) {
    RC.Type = RuntimeConstantType::INT32;
    RC.Value.Int32Val = *static_cast<const int32_t *>(Ptr);
  } else if (Ty->isIntegerTy(64)) {
    RC.Type = RuntimeConstantType::INT64;
    RC.Value.Int64Val = *static_cast<const int64_t *>(Ptr);
  } else if (Ty->isFloatTy()) {
    RC.Type = RuntimeConstantType::FLOAT;
    RC.Value.FloatVal = *static_cast<const float *>(Ptr);
  } else if (Ty->isDoubleTy()) {
    RC.Type = RuntimeConstantType::DOUBLE;
    RC.Value.DoubleVal = *static_cast<const double *>(Ptr);
  }

  return RC;
}

inline SmallVector<RuntimeConstant>
extractAutoDetectedCaptures(const void *LambdaClosure,
                            const SmallVector<CaptureInfo> &DetectedCaptures,
                            const DataLayout &DL, StructType *ClosureType) {
  SmallVector<RuntimeConstant> Result;
  if (!LambdaClosure || !ClosureType || DetectedCaptures.empty())
    return Result;

  const StructLayout *SL = DL.getStructLayout(ClosureType);
  const char *ClosureBytes = static_cast<const char *>(LambdaClosure);

  for (const auto &Cap : DetectedCaptures) {
    if (!Cap.IsReadOnly)
      continue;

    uint64_t ByteOffset = SL->getElementOffset(Cap.SlotIndex);
    Result.push_back(readValueFromMemory(ClosureBytes + ByteOffset,
                                         Cap.CaptureType, Cap.SlotIndex));
  }

  return Result;
}

inline StructType *inferClosureType(Function &F) {
  if (F.arg_empty())
    return nullptr;

  Argument *ClosureArg = &*F.arg_begin();
  for (User *U : ClosureArg->users()) {
    if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
      if (auto *STy = dyn_cast<StructType>(GEP->getSourceElementType()))
        return STy;
    }
  }

  return nullptr;
}

inline SmallString<128> traceOutAuto(int Slot, Constant *C) {
  SmallString<128> S;
  raw_svector_ostream OS(S);
  OS << "[LambdaSpec][Auto] Replacing slot " << Slot << " with " << *C << "\n";
  return S;
}

/// Overload for RuntimeConstant - formats value as LLVM type string
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
