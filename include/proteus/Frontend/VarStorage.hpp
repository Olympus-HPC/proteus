#ifndef PROTEUS_FRONTEND_VARSTORAGE_HPP
#define PROTEUS_FRONTEND_VARSTORAGE_HPP

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

namespace proteus {
using namespace llvm;

enum class StorageKind { Scalar, Pointer, Array };

class VarStorage {

protected:
  IRBuilderBase &IRB;

public:
  VarStorage(IRBuilderBase &IRB) : IRB(IRB) {}
  virtual ~VarStorage() = default;

  virtual StorageKind getKind() const = 0;

  virtual Value *getSlot() const = 0;
  // Load/store the logical value represented by this storage.
  virtual Value *loadValue() const = 0;
  virtual void storeValue(Value *Val) = 0;
  virtual Type *getAllocatedType() const = 0;
  virtual Type *getValueType() const = 0;
  virtual std::unique_ptr<VarStorage> clone() const = 0;
};

class ScalarStorage : public VarStorage {

  AllocaInst *Slot = nullptr;

public:
  ScalarStorage(AllocaInst *Slot, IRBuilderBase &IRB)
      : VarStorage(IRB), Slot(Slot) {}
  std::unique_ptr<VarStorage> clone() const override {
    return std::make_unique<ScalarStorage>(Slot, IRB);
  }

  StorageKind getKind() const override;
  Value *getSlot() const override;
  Value *loadValue() const override;
  void storeValue(Value *Val) override;
  Type *getAllocatedType() const override;
  Type *getValueType() const override;
};

class PointerStorage : public VarStorage {

  AllocaInst *PtrSlot = nullptr;
  Type *PointerElemTy = nullptr;

public:
  PointerStorage(AllocaInst *PtrSlot, IRBuilderBase &IRB, Type *PointerElemTy)
      : VarStorage(IRB), PtrSlot(PtrSlot), PointerElemTy(PointerElemTy) {}
  PointerStorage(Value *PtrSlot, IRBuilderBase &IRB, Type *PointerElemTy)
      : VarStorage(IRB), PtrSlot(dyn_cast<AllocaInst>(PtrSlot)),
        PointerElemTy(PointerElemTy) {}
  std::unique_ptr<VarStorage> clone() const override {
    return std::make_unique<PointerStorage>(PtrSlot, IRB, PointerElemTy);
  }

  StorageKind getKind() const override;
  Value *getSlot() const override;
  // Load/store the pointee value through the pointer stored in PtrSlot.
  Value *loadValue() const override;
  void storeValue(Value *Val) override;
  // Load/store the pointer value itself from/to PtrSlot.
  Value *loadPointer() const;
  void storePointer(Value *Ptr);
  Type *getAllocatedType() const override;
  Type *getValueType() const override;
};

class ArrayStorage : public VarStorage {

  Value *BasePointer = nullptr;
  ArrayType *ArrayTy = nullptr;

public:
  ArrayStorage(Value *BasePointer, IRBuilderBase &IRB, ArrayType *ArrayTy)
      : VarStorage(IRB), BasePointer(BasePointer), ArrayTy(ArrayTy) {}
  std::unique_ptr<VarStorage> clone() const override {
    return std::make_unique<ArrayStorage>(BasePointer, IRB, ArrayTy);
  }
  StorageKind getKind() const override;
  Value *getSlot() const override;
  Value *loadValue() const override;
  void storeValue(Value *Val) override;
  Type *getAllocatedType() const override;
  Type *getValueType() const override;
};

} // namespace proteus

#endif
