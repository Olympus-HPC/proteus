#include "proteus/Frontend/VarStorage.hpp"
#include "proteus/Error.h"

namespace proteus {

Value *ScalarStorage::getSlot() const { return Slot; }

Value *ScalarStorage::loadValue(AccessKind Kind) const {
  (void)Kind;
  return IRB.CreateLoad(Slot->getAllocatedType(), Slot);
}

void ScalarStorage::storeValue(Value *Val, AccessKind Kind) {
  (void)Kind;
  IRB.CreateStore(Val, Slot);
}

Type *ScalarStorage::getAllocatedType() const {
  return Slot->getAllocatedType();
}

Type *ScalarStorage::getValueType() const { return Slot->getAllocatedType(); }

Value *PointerStorage::getSlot() const { return PtrSlot; }

Value *PointerStorage::loadValue(AccessKind Kind) const {
  if (Kind == AccessKind::Direct)
    return IRB.CreateLoad(PtrSlot->getAllocatedType(), PtrSlot);

  Value *Ptr = IRB.CreateLoad(PtrSlot->getAllocatedType(), PtrSlot);
  return IRB.CreateLoad(PointerElemTy, Ptr);
}

void PointerStorage::storeValue(Value *Val, AccessKind Kind) {
  if (Kind == AccessKind::Direct) {
    IRB.CreateStore(Val, PtrSlot);
    return;
  }

  Value *Ptr = IRB.CreateLoad(PtrSlot->getAllocatedType(), PtrSlot);
  IRB.CreateStore(Val, Ptr);
}

Type *PointerStorage::getAllocatedType() const {
  return PtrSlot->getAllocatedType();
}

Type *PointerStorage::getValueType() const { return PointerElemTy; }

Value *ArrayStorage::getSlot() const { return BasePointer; }

Value *ArrayStorage::loadValue(AccessKind Kind) const {
  (void)Kind;
  PROTEUS_FATAL_ERROR("Cannot load entire array as a value");
}

void ArrayStorage::storeValue(Value *Val, AccessKind Kind) {
  (void)Val;
  (void)Kind;
  PROTEUS_FATAL_ERROR("Cannot store value to entire array");
}

Type *ArrayStorage::getAllocatedType() const { return ArrayTy; }

Type *ArrayStorage::getValueType() const { return ArrayTy->getElementType(); }

} // namespace proteus
