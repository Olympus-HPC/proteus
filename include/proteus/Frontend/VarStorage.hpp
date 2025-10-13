#ifndef PROTEUS_FRONTEND_VARSTORAGE_HPP
#define PROTEUS_FRONTEND_VARSTORAGE_HPP

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

namespace proteus {
using namespace llvm;

class VarStorage {

  protected:
  IRBuilderBase &IRB;

public:
  VarStorage(IRBuilderBase &IRB) : IRB(IRB) {}
  virtual ~VarStorage() = default;

  virtual Value *getValue() const = 0;
  virtual Value *getSlot() const = 0;
  virtual Value *loadValue() const = 0;
  virtual void storeValue(Value *Val) = 0;
  virtual Type *getAllocatedType() const = 0;
  virtual Type *getValueType() const = 0;

};

class ScalarStorage : public VarStorage {

  AllocaInst *Slot = nullptr;

  public:
  ScalarStorage(AllocaInst *Slot, IRBuilderBase &IRB) : VarStorage(IRB), Slot(Slot) {}

  Value *getSlot() const override;
  Value *getValue() const override;
  Value *loadValue() const override;
  void storeValue(Value *Val) override;
  Type *getAllocatedType() const override;
  Type *getValueType() const override;
};

class PointerStorage : public VarStorage {

  AllocaInst *PtrSlot = nullptr;
  Type *PointerElemTy = nullptr;

  public:
  PointerStorage(AllocaInst *PtrSlot, IRBuilderBase &IRB, Type *PointerElemTy) : VarStorage(IRB), PtrSlot(PtrSlot), PointerElemTy(PointerElemTy) {}
  PointerStorage(Value *PtrSlot, IRBuilderBase &IRB, Type *PointerElemTy) : VarStorage(IRB), PtrSlot(dyn_cast<AllocaInst>(PtrSlot)), PointerElemTy(PointerElemTy) {}

  Value *getValue() const override;
  Value *getSlot() const override;
  Value *loadValue() const override;
  void storeValue(Value *Val) override;
  Type *getAllocatedType() const override;
  Type *getValueType() const override;
  void storePointer(Value *Ptr);
  Value *getPointerValue() const;
};

class ArrayStorage : public VarStorage {

  Value *BasePointer = nullptr;
  ArrayType *ArrayTy = nullptr;

  public:
  ArrayStorage(Value *BasePointer, IRBuilderBase &IRB, ArrayType *ArrayTy) : VarStorage(IRB), BasePointer(BasePointer), ArrayTy(ArrayTy) {}

  Value *getValue() const override;
  Value *getSlot() const override;
  Value *loadValue() const override;
  void storeValue(Value *Val) override;
  Type *getAllocatedType() const override;
  Type *getValueType() const override;
};

}

#endif
