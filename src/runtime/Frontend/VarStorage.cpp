#include "proteus/Frontend/VarStorage.h"
#include "proteus/Error.h"
#include "proteus/Frontend/LLVMTypeMap.h"

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>

namespace proteus {

using namespace llvm;

// ---------------------------------------------------------------------------
// Local helper
// ---------------------------------------------------------------------------

static Type *getAllocaType(Value *SlotVal, const char *StorageKind) {
  AllocaInst *Alloca = dyn_cast<AllocaInst>(SlotVal);
  if (!Alloca)
    reportFatalError(std::string("Expected AllocaInst in ") + StorageKind);
  return Alloca->getAllocatedType();
}

// ---------------------------------------------------------------------------
// ScalarStorage
// ---------------------------------------------------------------------------

Value *ScalarStorage::getSlot() const { return Slot; }

Value *ScalarStorage::loadValue() const {
  return IRB.CreateLoad(getAllocaType(Slot, "ScalarStorage"), Slot);
}

void ScalarStorage::storeValue(Value *Val) { IRB.CreateStore(Val, Slot); }

IRType ScalarStorage::getAllocatedType() const { return IRValTy; }

IRType ScalarStorage::getValueType() const { return IRValTy; }

// ---------------------------------------------------------------------------
// PointerStorage
// ---------------------------------------------------------------------------

Value *PointerStorage::getSlot() const { return PtrSlot; }

// Load/store the pointee value through the pointer stored in PtrSlot.
Value *PointerStorage::loadValue() const {
  Type *ElemLLVMTy = toLLVMType(ElemIRTy, IRB.getContext());
  Value *Ptr =
      IRB.CreateLoad(getAllocaType(PtrSlot, "PointerStorage"), PtrSlot);
  return IRB.CreateLoad(ElemLLVMTy, Ptr);
}

void PointerStorage::storeValue(Value *Val) {
  Value *Ptr =
      IRB.CreateLoad(getAllocaType(PtrSlot, "PointerStorage"), PtrSlot);
  IRB.CreateStore(Val, Ptr);
}

// Load/store the pointer value itself from/to PtrSlot.
Value *PointerStorage::loadPointer() const {
  return IRB.CreateLoad(getAllocaType(PtrSlot, "PointerStorage"), PtrSlot);
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

IRType ArrayStorage::getAllocatedType() const {
  return IRType{IRTypeKind::Array, ElemIRTy.Signed, NElem, ElemIRTy.Kind};
}

IRType ArrayStorage::getValueType() const { return ElemIRTy; }

} // namespace proteus
