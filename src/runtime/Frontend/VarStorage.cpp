#include "proteus/Frontend/VarStorage.h"
#include "proteus/Error.h"
#include "proteus/Frontend/LLVMTypeMap.h"
#include "proteus/Frontend/LLVMValueMap.h"

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

IRValue ScalarStorage::getSlot() const { return Slot; }

IRValue ScalarStorage::loadValue() const {
  Value *LLVMSlot = toLLVMValue(Slot);
  return fromLLVMValue(
      IRB.CreateLoad(getAllocaType(LLVMSlot, "ScalarStorage"), LLVMSlot));
}

void ScalarStorage::storeValue(IRValue Val) {
  IRB.CreateStore(toLLVMValue(Val), toLLVMValue(Slot));
}

IRType ScalarStorage::getAllocatedType() const { return IRValTy; }

IRType ScalarStorage::getValueType() const { return IRValTy; }

// ---------------------------------------------------------------------------
// PointerStorage
// ---------------------------------------------------------------------------

IRValue PointerStorage::getSlot() const { return PtrSlot; }

// Load/store the pointee value through the pointer stored in PtrSlot.
IRValue PointerStorage::loadValue() const {
  Value *LLVMPtrSlot = toLLVMValue(PtrSlot);
  Type *ElemLLVMTy = toLLVMType(ElemIRTy, IRB.getContext());
  Value *Ptr =
      IRB.CreateLoad(getAllocaType(LLVMPtrSlot, "PointerStorage"), LLVMPtrSlot);
  return fromLLVMValue(IRB.CreateLoad(ElemLLVMTy, Ptr));
}

void PointerStorage::storeValue(IRValue Val) {
  Value *LLVMPtrSlot = toLLVMValue(PtrSlot);
  Value *Ptr =
      IRB.CreateLoad(getAllocaType(LLVMPtrSlot, "PointerStorage"), LLVMPtrSlot);
  IRB.CreateStore(toLLVMValue(Val), Ptr);
}

// Load/store the pointer value itself from/to PtrSlot.
IRValue PointerStorage::loadPointer() const {
  Value *LLVMPtrSlot = toLLVMValue(PtrSlot);
  return fromLLVMValue(IRB.CreateLoad(
      getAllocaType(LLVMPtrSlot, "PointerStorage"), LLVMPtrSlot));
}

void PointerStorage::storePointer(IRValue Val) {
  IRB.CreateStore(toLLVMValue(Val), toLLVMValue(PtrSlot));
}

IRType PointerStorage::getAllocatedType() const {
  return IRType{IRTypeKind::Pointer, ElemIRTy.Signed, 0, ElemIRTy.Kind};
}

IRType PointerStorage::getValueType() const { return ElemIRTy; }

// ---------------------------------------------------------------------------
// ArrayStorage
// ---------------------------------------------------------------------------

IRValue ArrayStorage::getSlot() const { return BasePointer; }

IRValue ArrayStorage::loadValue() const {
  reportFatalError("Cannot load entire array as a value");
}

void ArrayStorage::storeValue(IRValue Val) {
  (void)Val;
  reportFatalError("Cannot store value to entire array");
}

IRType ArrayStorage::getAllocatedType() const {
  return IRType{IRTypeKind::Array, ElemIRTy.Signed, NElem, ElemIRTy.Kind};
}

IRType ArrayStorage::getValueType() const { return ElemIRTy; }

} // namespace proteus
