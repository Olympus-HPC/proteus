//===-- AutoReadOnlyCaptures.cpp -- Find read-only lambda captures --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DESCRIPTION:
//    Find the scalar captures of a lambda call operator that are only loaded,
//    so they can be specialized without an explicit jit_variable.
//
//===----------------------------------------------------------------------===//

#include "AutoReadOnlyCaptures.h"
#include "Helpers.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>

#include <map>
#include <optional>

using namespace llvm;

namespace proteus {

namespace {

std::optional<RuntimeConstantType> getAutoCaptureType(Type *Ty) {
  if (Ty->isIntegerTy(8))
    return RuntimeConstantType::INT8;
  if (Ty->isIntegerTy(32))
    return RuntimeConstantType::INT32;
  if (Ty->isIntegerTy(64))
    return RuntimeConstantType::INT64;
  if (Ty->isFloatTy())
    return RuntimeConstantType::FLOAT;
  if (Ty->isDoubleTy())
    return RuntimeConstantType::DOUBLE;
  return std::nullopt;
}

std::optional<unsigned> getTopLevelSlot(GetElementPtrInst &GEP) {
  if (!isa<StructType>(GEP.getSourceElementType()) || GEP.getNumIndices() != 2)
    return std::nullopt;

  auto *First = dyn_cast<ConstantInt>(GEP.getOperand(1));
  auto *Second = dyn_cast<ConstantInt>(GEP.getOperand(2));
  if (!First || !First->isZero() || !Second)
    return std::nullopt;

  return Second->getZExtValue();
}

// Returns the type every user loads, or nullptr if some user is not a simple
// load or the loads disagree.
Type *getCommonLoadType(ArrayRef<User *> Users) {
  Type *LoadTy = nullptr;
  for (User *U : Users) {
    auto *LI = dyn_cast<LoadInst>(U);
    if (!LI || !LI->isSimple() || (LoadTy && LI->getType() != LoadTy)) {
      DEBUG(Logger::logs("proteus-pass")
            << "[AutoReadOnly] Slot not read-only: " << *U << "\n");
      return nullptr;
    }
    LoadTy = LI->getType();
  }
  return LoadTy;
}

} // namespace

SmallVector<AutoCapture, 4> analyzeAutoReadOnlyCaptures(Function &LambdaOp) {
  if (LambdaOp.isDeclaration() || LambdaOp.arg_empty())
    return {};

  // With an sret return the closure is not the first argument, which the
  // runtime lambda transform assumes.
  Argument *Closure = LambdaOp.getArg(0);
  if (!Closure->getType()->isPointerTy() || Closure->hasStructRetAttr())
    return {};

  // Slot 0 is often accessed through the closure pointer itself because a
  // (0, 0) struct GEP folds away.
  std::map<unsigned, SmallVector<User *, 4>> SlotUsers;
  StructType *ClosureTy = nullptr;
  for (User *U : Closure->users()) {
    if (isa<LoadInst>(U)) {
      SlotUsers[0].push_back(U);
      continue;
    }

    if (auto *SI = dyn_cast<StoreInst>(U);
        SI && SI->getValueOperand() != Closure) {
      SlotUsers[0].push_back(U);
      continue;
    }

    auto *GEP = dyn_cast<GetElementPtrInst>(U);
    auto Slot = GEP ? getTopLevelSlot(*GEP) : std::nullopt;
    auto *GEPTy =
        GEP ? dyn_cast<StructType>(GEP->getSourceElementType()) : nullptr;
    if (!Slot || (ClosureTy && GEPTy != ClosureTy)) {
      DEBUG(Logger::logs("proteus-pass")
            << "[AutoReadOnly] Closure escapes in " << LambdaOp.getName()
            << ": " << *U << "\n");
      return {};
    }

    ClosureTy = GEPTy;
    SlotUsers[*Slot].append(GEP->user_begin(), GEP->user_end());
  }

  const DataLayout &DL = LambdaOp.getParent()->getDataLayout();
  SmallVector<AutoCapture, 4> Captures;
  for (auto &[Slot, Users] : SlotUsers) {
    Type *LoadTy = getCommonLoadType(Users);
    if (!LoadTy)
      continue;

    auto RCType = getAutoCaptureType(LoadTy);
    if (!RCType)
      continue;

    uint32_t Offset = 0;
    if (ClosureTy) {
      if (ClosureTy->getElementType(Slot) != LoadTy)
        continue;
      Offset = DL.getStructLayout(ClosureTy)->getElementOffset(Slot);
    }

    Captures.push_back(AutoCapture{Slot, Offset, *RCType});
  }

  return Captures;
}

} // namespace proteus
