#include "proteus/Frontend/VarStorage.hpp"
#include "proteus/Error.h"

namespace proteus {

Value *ScalarStorage::loadValue() const {
  return IRB.CreateLoad(Slot->getAllocatedType(), Slot);
}

Value *ScalarStorage::getValue() const {
  return Slot;
}

void ScalarStorage::storeValue(Value *Val) {
  IRB.CreateStore(Val, Slot);
}

Type *ScalarStorage::getAllocatedType() const {
  return Slot->getAllocatedType();
}

Type *ScalarStorage::getValueType() const {
  return Slot->getAllocatedType();
}

Value *PointerStorage::loadValue() const {
  // Load the pointer from PtrSlot, then load the value from that pointer
  Value *Ptr = IRB.CreateLoad(PtrSlot->getAllocatedType(), PtrSlot);
  return IRB.CreateLoad(PointerElemTy, Ptr);
}

Value *PointerStorage::getValue() const {
  // Load and return the pointer (the address where the value lives)
  return IRB.CreateLoad(PtrSlot->getAllocatedType(), PtrSlot);
}

void PointerStorage::storeValue(Value *Val) {
  // Load the pointer, then store the value to that address
  Value *Ptr = IRB.CreateLoad(PtrSlot->getAllocatedType(), PtrSlot);
  IRB.CreateStore(Val, Ptr);
}

Type *PointerStorage::getAllocatedType() const {
  return PtrSlot->getAllocatedType();
}

Type *PointerStorage::getValueType() const {
  return PointerElemTy;
}

Value *ArrayStorage::getValue() const {
  return BasePointer;
}

Value *ArrayStorage::loadValue() const {
  PROTEUS_FATAL_ERROR("Cannot load entire array as a value");
}

void ArrayStorage::storeValue(Value *Val) {
  PROTEUS_FATAL_ERROR("Cannot store entire array as a value");
}

Type *ArrayStorage::getAllocatedType() const {
  return ArrayTy;
}

Type *ArrayStorage::getValueType() const {
  return ArrayTy->getElementType();
}

}
