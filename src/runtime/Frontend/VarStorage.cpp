#include "proteus/Frontend/VarStorage.h"
#include "proteus/Error.h"

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>

namespace proteus {

using namespace llvm;

// ---------------------------------------------------------------------------
// ScalarStorage
// ---------------------------------------------------------------------------

llvm::Type *ScalarStorage::llvmAllocaType() const {
  AllocaInst *Alloca = dyn_cast<AllocaInst>(Slot);
  if (!Alloca)
    reportFatalError("Expected AllocaInst in ScalarStorage");
  return Alloca->getAllocatedType();
}

Value *ScalarStorage::getSlot() const { return Slot; }

Value *ScalarStorage::loadValue() const {
  return IRB.CreateLoad(llvmAllocaType(), Slot);
}

void ScalarStorage::storeValue(Value *Val) { IRB.CreateStore(Val, Slot); }

Type *ScalarStorage::getSlotType() const { return Slot->getType(); }

IRType ScalarStorage::getAllocatedType() const { return IRValTy; }

IRType ScalarStorage::getValueType() const { return IRValTy; }

// ---------------------------------------------------------------------------
// PointerStorage
// ---------------------------------------------------------------------------

llvm::Type *PointerStorage::llvmAllocaType() const {
  AllocaInst *Alloca = dyn_cast<AllocaInst>(PtrSlot);
  if (!Alloca)
    reportFatalError("Expected AllocaInst in PointerStorage");
  return Alloca->getAllocatedType();
}

Value *PointerStorage::getSlot() const { return PtrSlot; }

Type *PointerStorage::getSlotType() const { return PtrSlot->getType(); }

// Load/store the pointee value through the pointer stored in PtrSlot.
Value *PointerStorage::loadValue() const {
  Value *Ptr = IRB.CreateLoad(llvmAllocaType(), PtrSlot);
  return IRB.CreateLoad(PointerElemTy, Ptr);
}

void PointerStorage::storeValue(Value *Val) {
  Value *Ptr = IRB.CreateLoad(llvmAllocaType(), PtrSlot);
  IRB.CreateStore(Val, Ptr);
}

// Load/store the pointer value itself from/to PtrSlot.
Value *PointerStorage::loadPointer() const {
  return IRB.CreateLoad(llvmAllocaType(), PtrSlot);
}

void PointerStorage::storePointer(Value *Val) { IRB.CreateStore(Val, PtrSlot); }

IRType PointerStorage::getAllocatedType() const {
  return IRType{IRTypeKind::Pointer, ElemIRTy.Signed, 0, ElemIRTy.Kind};
}

IRType PointerStorage::getValueType() const { return ElemIRTy; }

// ---------------------------------------------------------------------------
// ArrayStorage
// ---------------------------------------------------------------------------

Value *ArrayStorage::getSlot() const { return BasePointer; }

Value *ArrayStorage::loadValue() const {
  reportFatalError("Cannot load entire array as a value");
}

void ArrayStorage::storeValue(Value *Val) {
  (void)Val;
  reportFatalError("Cannot store value to entire array");
}

Type *ArrayStorage::getSlotType() const { return BasePointer->getType(); }

IRType ArrayStorage::getAllocatedType() const {
  return IRType{IRTypeKind::Array, ElemIRTy.Signed, ArrayTy->getNumElements(),
                ElemIRTy.Kind};
}

IRType ArrayStorage::getValueType() const { return ElemIRTy; }

} // namespace proteus
