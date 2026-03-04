#include "AutoReadOnlyCapturesAnalysis.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>

#include <optional>

#include <cassert>

using namespace llvm;

namespace proteus {
namespace {

constexpr char AutoReadOnlyCapturesMetadataName[] =
    "proteus.auto_readonly_captures";

struct ParsedCaptureAccess {
  int32_t SlotIndex;
  int32_t ByteOffset;
};

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

RuntimeConstantType classifySupportedScalar(Type *Ty) {
  if (auto *IT = dyn_cast<IntegerType>(Ty)) {
    switch (IT->getBitWidth()) {
    case 1:
      return RuntimeConstantType::BOOL;
    case 8:
      return RuntimeConstantType::INT8;
    case 32:
      return RuntimeConstantType::INT32;
    case 64:
      return RuntimeConstantType::INT64;
    default:
      return RuntimeConstantType::NONE;
    }
  }

  if (Ty->isFloatTy())
    return RuntimeConstantType::FLOAT;

  if (Ty->isDoubleTy())
    return RuntimeConstantType::DOUBLE;

  return RuntimeConstantType::NONE;
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

bool pointerEscapes(Value *RootPtr) {
  assert(RootPtr->getType()->isPointerTy() && "Expected pointer root");

  SmallVector<Value *, 16> WorkList{RootPtr};
  SmallPtrSet<Value *, 32> Seen;

  while (!WorkList.empty()) {
    Value *V = WorkList.pop_back_val();
    if (!Seen.insert(V).second)
      continue;

    for (User *U : V->users()) {
      PointerUseEffect Effect = classifyPointerUse(U, V);
      if (Effect == PointerUseEffect::ReadOnly ||
          Effect == PointerUseEffect::Ignore)
        continue;

      if (Effect == PointerUseEffect::BenignTransform) {
        WorkList.push_back(cast<Value>(U));
        continue;
      }

      return true;
    }
  }

  return false;
}

std::optional<ParsedCaptureAccess> parseGEPAccess(Function &F,
                                                  GetElementPtrInst *GEP) {
  if (GEP->getNumIndices() < 2)
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
  assert(SlotIndex >= 0 && "Expected constant slot index");

  if (static_cast<unsigned>(SlotIndex) >= SourceStruct->getNumElements())
    return std::nullopt;

  const DataLayout &DL = F.getParent()->getDataLayout();
  const StructLayout *SL = DL.getStructLayout(SourceStruct);
  int32_t ByteOffset = static_cast<int32_t>(SL->getElementOffset(SlotIndex));
  return ParsedCaptureAccess{SlotIndex, ByteOffset};
}

std::optional<AutoReadOnlyCaptureMetadataEntry>
parseDirectLoadCapture(LoadInst *LI) {
  RuntimeConstantType RCType = classifySupportedScalar(LI->getType());
  if (RCType == RuntimeConstantType::NONE)
    return std::nullopt;

  return AutoReadOnlyCaptureMetadataEntry{/*SlotIndex=*/0,
                                          /*ByteOffset=*/0, RCType};
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

  if (State.RCType == RuntimeConstantType::NONE)
    State.RCType = Entry.RCType;
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

void analyzePointerUsersForSlot(Value *RootPtr, int32_t SlotIndex,
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

        auto Parsed = parseDirectLoadCapture(LI);
        if (!Parsed)
          continue;

        Parsed->SlotIndex = SlotIndex;
        Parsed->ByteOffset = ByteOffset;
        updateReadOnlySlot(Slots, *Parsed);
        continue;
      }

      PointerUseEffect Effect = classifyPointerUse(U, V);
      if (Effect == PointerUseEffect::Ignore ||
          Effect == PointerUseEffect::ReadOnly)
        continue;

      if (Effect == PointerUseEffect::BenignTransform) {
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

        auto Parsed = parseDirectLoadCapture(LI);
        if (!Parsed)
          continue;

        updateReadOnlySlot(Slots, *Parsed);
        continue;
      }

      if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
        auto Parsed = parseGEPAccess(F, GEP);
        if (!Parsed)
          continue;

        assert(Parsed->SlotIndex >= 0 && "Expected constant slot index");
        analyzePointerUsersForSlot(GEP, Parsed->SlotIndex, Parsed->ByteOffset,
                                   Slots);
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
