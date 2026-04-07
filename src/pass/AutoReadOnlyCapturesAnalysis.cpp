#include "AutoReadOnlyCapturesAnalysis.h"

#include "proteus/impl/RuntimeConstantTypeHelpers.h"

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/CaptureTracking.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>

#include <limits>
#include <optional>

#include <cassert>

using namespace llvm;

namespace proteus {
namespace {

constexpr char AutoReadOnlyCapturesMetadataName[] =
    "proteus.auto_readonly_captures";

struct SlotState {
  int32_t ByteOffset;
  RuntimeConstantType RCType = RuntimeConstantType::NONE;
  bool IsReadOnly = true;
};

enum class PointerUseEffect {
  Ignore,
  ReadOnly,
  BenignTransform,
  WriteOrEscape,
};

std::optional<RuntimeConstantType> classifySupportedScalar(Type *Ty) {
  if (!(Ty->isIntegerTy(1) || Ty->isIntegerTy(8) || Ty->isIntegerTy(32) ||
        Ty->isIntegerTy(64) || Ty->isFloatTy() || Ty->isDoubleTy())) {
    return std::nullopt;
  }

  RuntimeConstantType RCType = convertTypeToRuntimeConstantType(Ty);
  if (!isSupportedAutoReadOnlyRCType(RCType))
    return std::nullopt;
  return RCType;
}

std::optional<int32_t> getGEPByteOffset(Function &F, GetElementPtrInst *GEP) {
  const DataLayout &DL = F.getParent()->getDataLayout();
  APInt Offset(DL.getPointerTypeSizeInBits(GEP->getType()), 0, true);
  if (!GEP->accumulateConstantOffset(DL, Offset))
    return std::nullopt;

  int64_t ByteOffset64 = Offset.getSExtValue();
  if (ByteOffset64 < 0 || ByteOffset64 > std::numeric_limits<int32_t>::max())
    return std::nullopt;

  return static_cast<int32_t>(ByteOffset64);
}

std::optional<int32_t> getTopLevelSlotIndex(Argument *ClosureArg,
                                            Value *BasePtr,
                                            GetElementPtrInst *GEP) {
  if (BasePtr != ClosureArg || GEP->getNumIndices() != 2)
    return std::nullopt;

  auto *SourceStruct = dyn_cast<StructType>(GEP->getSourceElementType());
  if (!SourceStruct)
    return std::nullopt;

  auto *SlotIdxConst = dyn_cast<ConstantInt>(GEP->getOperand(2));
  if (!SlotIdxConst)
    return std::nullopt;

  int64_t SlotIdx64 = SlotIdxConst->getSExtValue();
  if (SlotIdx64 < 0)
    return std::nullopt;

  int32_t SlotIndex = static_cast<int32_t>(SlotIdx64);
  if (static_cast<int64_t>(SlotIndex) != SlotIdx64)
    return std::nullopt;

  if (static_cast<unsigned>(SlotIndex) >= SourceStruct->getNumElements())
    return std::nullopt;

  if (!classifySupportedScalar(GEP->getResultElementType()))
    return std::nullopt;

  return SlotIndex;
}

PointerUseEffect classifyPointerUse(User *U, Value *Ptr) {
  if (auto *LI = dyn_cast<LoadInst>(U)) {
    if (LI->getPointerOperand() == Ptr)
      return PointerUseEffect::ReadOnly;
    return PointerUseEffect::WriteOrEscape;
  }

  if (auto *SI = dyn_cast<StoreInst>(U)) {
    if (SI->getPointerOperand() == Ptr)
      return PointerUseEffect::WriteOrEscape;
    if (SI->getValueOperand() == Ptr)
      return PointerUseEffect::WriteOrEscape;
    return PointerUseEffect::Ignore;
  }

  if (isa<CallBase>(U))
    return PointerUseEffect::WriteOrEscape;

  if (isa<GetElementPtrInst>(U) || isa<BitCastInst>(U) ||
      isa<AddrSpaceCastInst>(U) || isa<PHINode>(U) || isa<SelectInst>(U))
    return PointerUseEffect::BenignTransform;

  return PointerUseEffect::WriteOrEscape;
}

class ClosureEscapeTracker final : public CaptureTracker {
  bool Captured = false;

public:
  bool pointerEscapes() const { return Captured; }

  void tooManyUses() override { Captured = true; }

  bool shouldExplore(const Use *U) override {
    // Field accesses are analyzed per-slot below. Escapes through a field GEP
    // should disqualify only that slot, not the whole closure object.
    if (auto *GEP = dyn_cast<GetElementPtrInst>(U->getUser());
        GEP && GEP->getPointerOperand() == U->get()) {
      return false;
    }

    return true;
  }

  bool captured(const Use *U) override {
    Captured = true;
    return true;
  }
};

bool pointerEscapes(Value *RootPtr) {
  assert(RootPtr->getType()->isPointerTy() && "Expected pointer root");

  ClosureEscapeTracker Tracker;
  PointerMayBeCaptured(RootPtr, &Tracker);
  return Tracker.pointerEscapes();
}

std::optional<AutoReadOnlyCaptureMetadataEntry>
parseDirectLoadCapture(LoadInst *LI, int32_t SlotIndex, int32_t ByteOffset) {
  auto RCType = classifySupportedScalar(LI->getType());
  if (!RCType)
    return std::nullopt;

  return AutoReadOnlyCaptureMetadataEntry{SlotIndex, ByteOffset, *RCType};
}

void updateReadOnlySlot(DenseMap<int32_t, SlotState> &Slots,
                        const AutoReadOnlyCaptureMetadataEntry &Entry) {
  auto It = Slots.find(Entry.SlotIndex);
  if (It == Slots.end()) {
    SlotState NewState;
    NewState.ByteOffset = Entry.ByteOffset;
    NewState.RCType = Entry.RCType;
    Slots.insert({Entry.SlotIndex, NewState});
    return;
  }

  SlotState &State = It->second;
  if (!State.IsReadOnly)
    return;

  if (State.ByteOffset != Entry.ByteOffset) {
    State.IsReadOnly = false;
    return;
  }

  if (State.RCType == RuntimeConstantType::NONE) {
    State.RCType = Entry.RCType;
    return;
  }

  if (State.RCType != Entry.RCType)
    State.IsReadOnly = false;
}

void markSlotNonReadOnly(DenseMap<int32_t, SlotState> &Slots, int32_t SlotIdx,
                         int32_t ByteOffset) {
  auto It = Slots.find(SlotIdx);
  if (It == Slots.end()) {
    SlotState NewState;
    NewState.ByteOffset = ByteOffset;
    NewState.IsReadOnly = false;
    Slots.insert({SlotIdx, NewState});
    return;
  }

  It->second.IsReadOnly = false;
}

void analyzePointerUsersForSlot(Function &F, Argument *ClosureArg,
                                Value *RootPtr, int32_t SlotIndex,
                                int32_t ByteOffset,
                                DenseMap<int32_t, SlotState> &Slots) {
  auto Existing = Slots.find(SlotIndex);
  if (Existing != Slots.end() && !Existing->second.IsReadOnly)
    return;

  SmallVector<Value *, 16> WorkList{RootPtr};
  SmallPtrSet<Value *, 32> Seen;

  while (!WorkList.empty()) {
    Value *V = WorkList.pop_back_val();
    if (!Seen.insert(V).second)
      continue;

    for (User *U : V->users()) {
      if (auto *LI = dyn_cast<LoadInst>(U)) {
        if (LI->getPointerOperand() != V) {
          markSlotNonReadOnly(Slots, SlotIndex, ByteOffset);
          break;
        }

        auto Parsed = parseDirectLoadCapture(LI, SlotIndex, ByteOffset);
        if (!Parsed)
          continue;

        updateReadOnlySlot(Slots, *Parsed);
        continue;
      }

      if (auto *SI = dyn_cast<StoreInst>(U)) {
        if (SI->getPointerOperand() == V || SI->getValueOperand() == V) {
          markSlotNonReadOnly(Slots, SlotIndex, ByteOffset);
          break;
        }
        continue;
      }

      PointerUseEffect Effect = classifyPointerUse(U, V);
      if (Effect == PointerUseEffect::Ignore ||
          Effect == PointerUseEffect::ReadOnly)
        continue;

      if (Effect == PointerUseEffect::BenignTransform) {
        if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
          auto LocalOffset = getGEPByteOffset(F, GEP);
          if (!LocalOffset) {
            markSlotNonReadOnly(Slots, SlotIndex, ByteOffset);
            break;
          }

          int32_t NestedByteOffset = ByteOffset + *LocalOffset;
          auto NestedSlot = getTopLevelSlotIndex(ClosureArg, V, GEP);
          analyzePointerUsersForSlot(F, ClosureArg, GEP,
                                     NestedSlot.value_or(NestedByteOffset),
                                     NestedByteOffset, Slots);
          continue;
        }
        WorkList.push_back(cast<Value>(U));
        continue;
      }

      markSlotNonReadOnly(Slots, SlotIndex, ByteOffset);
      break;
    }

    auto It = Slots.find(SlotIndex);
    if (It != Slots.end() && !It->second.IsReadOnly)
      break;
  }
}

SmallVector<AutoReadOnlyCaptureMetadataEntry>
collectReadOnlyCaptures(const DenseMap<int32_t, SlotState> &Slots) {
  SmallVector<AutoReadOnlyCaptureMetadataEntry> Captures;

  for (const auto &Entry : Slots) {
    int32_t SlotIndex = Entry.first;
    const SlotState &State = Entry.second;
    if (!State.IsReadOnly)
      continue;
    if (State.RCType == RuntimeConstantType::NONE)
      continue;

    Captures.push_back(AutoReadOnlyCaptureMetadataEntry{
        SlotIndex, State.ByteOffset, State.RCType});
  }

  llvm::sort(Captures, [](const AutoReadOnlyCaptureMetadataEntry &L,
                          const AutoReadOnlyCaptureMetadataEntry &R) {
    return L.SlotIndex < R.SlotIndex;
  });

  return Captures;
}

} // namespace

SmallVector<AutoReadOnlyCaptureMetadataEntry>
analyzeAutoReadOnlyCaptures(Function &F) {
  SmallVector<AutoReadOnlyCaptureMetadataEntry> Captures;

  if (F.arg_empty())
    return Captures;

  Argument *ClosureArg = &*F.arg_begin();
  if (!ClosureArg->getType()->isPointerTy())
    return Captures;

  if (pointerEscapes(ClosureArg))
    return Captures;

  DenseMap<int32_t, SlotState> Slots;

  SmallVector<Value *, 16> WorkList{ClosureArg};
  SmallPtrSet<Value *, 32> Seen;

  while (!WorkList.empty()) {
    Value *V = WorkList.pop_back_val();
    if (!Seen.insert(V).second)
      continue;

    for (User *U : V->users()) {
      if (auto *LI = dyn_cast<LoadInst>(U)) {
        if (LI->getPointerOperand() != V)
          continue;

        auto Parsed =
            parseDirectLoadCapture(LI, /*SlotIndex=*/0, /*ByteOffset=*/0);
        if (!Parsed)
          continue;

        updateReadOnlySlot(Slots, *Parsed);
        continue;
      }

      if (auto *SI = dyn_cast<StoreInst>(U)) {
        if (SI->getPointerOperand() != V)
          continue;

        // A direct store through the closure pointer writes the first slot.
        markSlotNonReadOnly(Slots, /*SlotIdx=*/0, /*ByteOffset=*/0);
        continue;
      }

      if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
        auto LocalOffset = getGEPByteOffset(F, GEP);
        if (!LocalOffset)
          continue;

        auto TopLevelSlot = getTopLevelSlotIndex(ClosureArg, V, GEP);
        int32_t ByteOffset = *LocalOffset;
        analyzePointerUsersForSlot(F, ClosureArg, GEP,
                                   TopLevelSlot.value_or(ByteOffset),
                                   ByteOffset, Slots);
        continue;
      }

      if (isa<BitCastInst>(U) || isa<AddrSpaceCastInst>(U) || isa<PHINode>(U) ||
          isa<SelectInst>(U)) {
        WorkList.push_back(cast<Value>(U));
        continue;
      }
    }
  }

  return collectReadOnlyCaptures(Slots);
}

void emitAutoReadOnlyCapturesMetadata(
    Function &F, ArrayRef<AutoReadOnlyCaptureMetadataEntry> Captures) {
  if (Captures.empty()) {
    F.setMetadata(AutoReadOnlyCapturesMetadataName, nullptr);
    return;
  }

  LLVMContext &Ctx = F.getContext();
  Type *I32Ty = Type::getInt32Ty(Ctx);

  SmallVector<Metadata *> Entries;
  Entries.reserve(Captures.size());
  for (const auto &Capture : Captures) {
    Entries.push_back(MDNode::get(
        Ctx,
        {ConstantAsMetadata::get(ConstantInt::get(I32Ty, Capture.SlotIndex)),
         ConstantAsMetadata::get(ConstantInt::get(I32Ty, Capture.ByteOffset)),
         ConstantAsMetadata::get(
             ConstantInt::get(I32Ty, static_cast<int32_t>(Capture.RCType)))}));
  }

  F.setMetadata(AutoReadOnlyCapturesMetadataName, MDNode::get(Ctx, Entries));
}

void annotateAutoReadOnlyCaptures(Module &M) {
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;

    auto Captures = analyzeAutoReadOnlyCaptures(F);
    emitAutoReadOnlyCapturesMetadata(F, Captures);
  }
}

} // namespace proteus
