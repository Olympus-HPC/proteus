#include "proteus/Frontend/VarStorage.hpp"
#include "proteus/Error.h"

namespace proteus {

Value *ScalarStorage::getSlot() const { return Slot; }

Value *ScalarStorage::loadValue() const {
  return IRB.CreateLoad(Slot->getAllocatedType(), Slot);
}

void ScalarStorage::storeValue(Value *Val) { IRB.CreateStore(Val, Slot); }

Type *ScalarStorage::getAllocatedType() const {
  return Slot->getAllocatedType();
}

Type *ScalarStorage::getValueType() const { return Slot->getAllocatedType(); }

Value *PointerStorage::getSlot() const { return PtrSlot; }

// Load/store the pointee value through the pointer stored in PtrSlot.
Value *PointerStorage::loadValue() const {
  Value *Ptr = IRB.CreateLoad(PtrSlot->getAllocatedType(), PtrSlot);
  return IRB.CreateLoad(PointerElemTy, Ptr);
}

void PointerStorage::storeValue(Value *Val) {
  Value *Ptr = IRB.CreateLoad(PtrSlot->getAllocatedType(), PtrSlot);
  IRB.CreateStore(Val, Ptr);
}

// Load/store the pointer value itself from/to PtrSlot.
Value *PointerStorage::loadPointer() const {
  return IRB.CreateLoad(PtrSlot->getAllocatedType(), PtrSlot);
}

void PointerStorage::storePointer(Value *Val) { IRB.CreateStore(Val, PtrSlot); }

Type *PointerStorage::getAllocatedType() const {
  return PtrSlot->getAllocatedType();
}

Type *PointerStorage::getValueType() const { return PointerElemTy; }

Value *ArrayStorage::getSlot() const { return BasePointer; }

Value *ArrayStorage::loadValue() const {
  PROTEUS_FATAL_ERROR("Cannot load entire array as a value");
}

void ArrayStorage::storeValue(Value *Val) {
  (void)Val;
  PROTEUS_FATAL_ERROR("Cannot store value to entire array");
}

Type *ArrayStorage::getAllocatedType() const { return ArrayTy; }

Type *ArrayStorage::getValueType() const { return ArrayTy->getElementType(); }

} // namespace proteus
