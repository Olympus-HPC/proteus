#include "proteus/Frontend/VarStorage.h"
#include "proteus/Error.h"

#include <llvm/IR/IRBuilder.h>

namespace proteus {

using namespace llvm;

Value *ScalarStorage::getSlot() const { return Slot; }

Value *ScalarStorage::loadValue() const {
  return IRB.CreateLoad(getAllocatedType(), Slot);
}

void ScalarStorage::storeValue(Value *Val) { IRB.CreateStore(Val, Slot); }

Type *ScalarStorage::getSlotType() const { return Slot->getType(); }

Type *ScalarStorage::getAllocatedType() const {
  AllocaInst *Alloca = dyn_cast<AllocaInst>(Slot);
  if (!Alloca)
    reportFatalError("Expected AllocaInst in ScalarStorage::loadValue");
  return Alloca->getAllocatedType();
}

Type *ScalarStorage::getValueType() const { return getAllocatedType(); }

Value *PointerStorage::getSlot() const { return PtrSlot; }

Type *PointerStorage::getSlotType() const { return PtrSlot->getType(); }

// Load/store the pointee value through the pointer stored in PtrSlot.
Value *PointerStorage::loadValue() const {
  Value *Ptr = IRB.CreateLoad(getAllocatedType(), PtrSlot);
  return IRB.CreateLoad(PointerElemTy, Ptr);
}

void PointerStorage::storeValue(Value *Val) {
  Value *Ptr = IRB.CreateLoad(getAllocatedType(), PtrSlot);
  IRB.CreateStore(Val, Ptr);
}

// Load/store the pointer value itself from/to PtrSlot.
Value *PointerStorage::loadPointer() const {
  return IRB.CreateLoad(getAllocatedType(), PtrSlot);
}

void PointerStorage::storePointer(Value *Val) { IRB.CreateStore(Val, PtrSlot); }

Type *PointerStorage::getAllocatedType() const {
  AllocaInst *Alloca = dyn_cast<AllocaInst>(PtrSlot);
  if (!Alloca)
    reportFatalError("Expected AllocaInst in PointerStorage::getAllocatedType");
  return Alloca->getAllocatedType();
}

Type *PointerStorage::getValueType() const { return PointerElemTy; }

Value *ArrayStorage::getSlot() const { return BasePointer; }

Value *ArrayStorage::loadValue() const {
  reportFatalError("Cannot load entire array as a value");
}

void ArrayStorage::storeValue(Value *Val) {
  (void)Val;
  reportFatalError("Cannot store value to entire array");
}
Type *ArrayStorage::getSlotType() const { return BasePointer->getType(); }

Type *ArrayStorage::getAllocatedType() const { return ArrayTy; }

Type *ArrayStorage::getValueType() const { return ArrayTy->getElementType(); }

} // namespace proteus
