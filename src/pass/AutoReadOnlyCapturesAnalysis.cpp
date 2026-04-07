#include "AutoReadOnlyCapturesAnalysis.h"

#include "Helpers.h"
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
  if (!GEP->accumulateConstantOffset(DL, Offset)) {
    DEBUG(Logger::logs("proteus-pass")
          << "[AutoReadOnly] Cannot compute constant GEP offset: " << *GEP
          << "\n");
    return std::nullopt;
  }

  int64_t ByteOffset64 = Offset.getSExtValue();
  if (ByteOffset64 < 0 || ByteOffset64 > std::numeric_limits<int32_t>::max()) {
    DEBUG(Logger::logs("proteus-pass")
          << "[AutoReadOnly] GEP offset out of range: " << ByteOffset64
          << " for GEP: " << *GEP << "\n");
    return std::nullopt;
  }

  return static_cast<int32_t>(ByteOffset64);
}

std::optional<int32_t> getTopLevelSlotIndex(Argument *ClosureArg,
                                            Value *BasePtr,
                                            GetElementPtrInst *GEP) {
  if (BasePtr != ClosureArg || GEP->getNumIndices() != 2) {
    DEBUG(Logger::logs("proteus-pass")
          << "[AutoReadOnly] GEP not top-level closure access: " << *GEP
          << "\n");
    return std::nullopt;
  }

  auto *SourceStruct = dyn_cast<StructType>(GEP->getSourceElementType());
  if (!SourceStruct) {
    DEBUG(Logger::logs("proteus-pass")
          << "[AutoReadOnly] GEP source not a struct: " << *GEP << "\n");
    return std::nullopt;
  }

  auto *SlotIdxConst = dyn_cast<ConstantInt>(GEP->getOperand(2));
  if (!SlotIdxConst) {
    DEBUG(Logger::logs("proteus-pass")
          << "[AutoReadOnly] GEP slot index not constant: " << *GEP << "\n");
    return std::nullopt;
  }

  int64_t SlotIdx64 = SlotIdxConst->getSExtValue();
  if (SlotIdx64 < 0) {
    DEBUG(Logger::logs("proteus-pass")
          << "[AutoReadOnly] GEP slot index negative: " << SlotIdx64
          << " in: " << *GEP << "\n");
    return std::nullopt;
  }

  int32_t SlotIndex = static_cast<int32_t>(SlotIdx64);
  if (static_cast<int64_t>(SlotIndex) != SlotIdx64) {
    DEBUG(Logger::logs("proteus-pass")
          << "[AutoReadOnly] GEP slot index overflow: " << SlotIdx64
          << " in: " << *GEP << "\n");
    return std::nullopt;
  }

  if (static_cast<unsigned>(SlotIndex) >= SourceStruct->getNumElements()) {
    DEBUG(Logger::logs("proteus-pass")
          << "[AutoReadOnly] GEP slot index out of bounds: " << SlotIndex
          << " >= " << SourceStruct->getNumElements() << " in: " << *GEP
          << "\n");
    return std::nullopt;
  }

  if (!classifySupportedScalar(GEP->getResultElementType())) {
    DEBUG(Logger::logs("proteus-pass")
          << "[AutoReadOnly] Unsupported GEP result type: "
          << *GEP->getResultElementType() << " in: " << *GEP << "\n");
    return std::nullopt;
  }

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

  void tooManyUses() override {
    DEBUG(Logger::logs("proteus-pass")
          << "[AutoReadOnly] Closure has too many uses for tracking\n");
    Captured = true;
  }

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
    DEBUG(Logger::logs("proteus-pass")
          << "[AutoReadOnly] Closure escapes through: " << *U->getUser()
          << "\n");
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
  if (!RCType) {
    DEBUG(Logger::logs("proteus-pass")
          << "[AutoReadOnly] Unsupported load type: " << *LI->getType()
          << " in: " << *LI << "\n");
    return std::nullopt;
  }

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
    DEBUG(Logger::logs("proteus-pass")
          << "[AutoReadOnly] Slot " << Entry.SlotIndex
          << " disqualified: inconsistent byte offsets (" << State.ByteOffset
          << " vs " << Entry.ByteOffset << ")\n");
    State.IsReadOnly = false;
    return;
  }

  if (State.RCType == RuntimeConstantType::NONE) {
    State.RCType = Entry.RCType;
    return;
  }

  if (State.RCType != Entry.RCType) {
    DEBUG(Logger::logs("proteus-pass")
          << "[AutoReadOnly] Slot " << Entry.SlotIndex
          << " disqualified: inconsistent types (" << toString(State.RCType)
          << " vs " << toString(Entry.RCType) << ")\n");
    State.IsReadOnly = false;
  }
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

  DEBUG(Logger::logs("proteus-pass")
        << "[AutoReadOnly] Analyzing slot " << SlotIndex << " at offset "
        << ByteOffset << "\n");

  SmallVector<Value *, 16> WorkList{RootPtr};
  SmallPtrSet<Value *, 32> Seen;

  while (!WorkList.empty()) {
    Value *V = WorkList.pop_back_val();
    if (!Seen.insert(V).second)
      continue;

    for (User *U : V->users()) {
      if (auto *LI = dyn_cast<LoadInst>(U)) {
        if (LI->getPointerOperand() != V) {
          DEBUG(Logger::logs("proteus-pass")
                << "[AutoReadOnly] Slot " << SlotIndex
                << " disqualified: pointer used as value in load: " << *LI
                << "\n");
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
          DEBUG(Logger::logs("proteus-pass")
                << "[AutoReadOnly] Slot " << SlotIndex
                << " disqualified: written by store: " << *SI << "\n");
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
            DEBUG(Logger::logs("proteus-pass")
                  << "[AutoReadOnly] Slot " << SlotIndex
                  << " disqualified: non-constant nested GEP: " << *GEP
                  << "\n");
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

      if (Effect == PointerUseEffect::WriteOrEscape) {
        DEBUG(Logger::logs("proteus-pass")
              << "[AutoReadOnly] Slot " << SlotIndex
              << " disqualified: write/escape use: " << *U << "\n");
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

  DEBUG(Logger::logs("proteus-pass")
        << "[AutoReadOnly] Analyzing function: " << F.getName() << "\n");

  if (F.arg_empty()) {
    DEBUG(Logger::logs("proteus-pass")
          << "[AutoReadOnly] Skipping " << F.getName() << ": no arguments\n");
    return Captures;
  }

  Argument *ClosureArg = &*F.arg_begin();
  if (!ClosureArg->getType()->isPointerTy()) {
    DEBUG(Logger::logs("proteus-pass")
          << "[AutoReadOnly] Skipping " << F.getName()
          << ": first argument not a pointer (type: " << *ClosureArg->getType()
          << ")\n");
    return Captures;
  }

  if (pointerEscapes(ClosureArg)) {
    DEBUG(Logger::logs("proteus-pass")
          << "[AutoReadOnly] Function " << F.getName()
          << ": closure pointer escapes\n");
    return Captures;
  }

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
        DEBUG(Logger::logs("proteus-pass")
              << "[AutoReadOnly] Slot 0 disqualified: direct store to closure: "
              << *SI << "\n");
        continue;
      }

      if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
        auto LocalOffset = getGEPByteOffset(F, GEP);
        if (!LocalOffset) {
          DEBUG(Logger::logs("proteus-pass")
                << "[AutoReadOnly] Skipping GEP with non-constant offset: "
                << *GEP << "\n");
          continue;
        }

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

  auto Result = collectReadOnlyCaptures(Slots);

  DEBUG(Logger::logs("proteus-pass")
        << "[AutoReadOnly] Function " << F.getName() << ": " << Slots.size()
        << " slots analyzed, " << Result.size() << " qualified as readonly\n");

  if (!Result.empty()) {
    DEBUG(Logger::logs("proteus-pass") << "[AutoReadOnly] Qualified slots:");
    for (const auto &Cap : Result) {
      DEBUG(Logger::logs("proteus-pass")
            << " [slot " << Cap.SlotIndex << " @ offset " << Cap.ByteOffset
            << ", type " << toString(Cap.RCType) << "]");
    }
    DEBUG(Logger::logs("proteus-pass") << "\n");
  }

  return Result;
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
