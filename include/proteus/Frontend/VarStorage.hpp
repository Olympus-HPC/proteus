#ifndef PROTEUS_FRONTEND_VARSTORAGE_HPP
#define PROTEUS_FRONTEND_VARSTORAGE_HPP

#include <memory>

namespace llvm {
class Value;
class IRBuilderBase;
class Type;
class ArrayType;
} // namespace llvm

namespace proteus {
using namespace llvm;

class VarStorage {

protected:
  IRBuilderBase &IRB;

public:
  VarStorage(IRBuilderBase &IRB) : IRB(IRB) {}
  virtual ~VarStorage() = default;

  virtual Value *getSlot() const = 0;
  // Load/store the logical value represented by this storage.
  virtual Value *loadValue() const = 0;
  virtual void storeValue(Value *Val) = 0;
  virtual Type *getSlotType() const = 0;
  virtual Type *getAllocatedType() const = 0;
  virtual Type *getValueType() const = 0;
  virtual std::unique_ptr<VarStorage> clone() const = 0;
};

class ScalarStorage : public VarStorage {
  Value *Slot = nullptr;

public:
  ScalarStorage(Value *Slot, IRBuilderBase &IRB)
      : VarStorage(IRB), Slot(Slot) {}
  std::unique_ptr<VarStorage> clone() const override {
    return std::make_unique<ScalarStorage>(Slot, IRB);
  }

  Value *getSlot() const override;
  Value *loadValue() const override;
  void storeValue(Value *Val) override;
  Type *getSlotType() const override;
  Type *getAllocatedType() const override;
  Type *getValueType() const override;
};

class PointerStorage : public VarStorage {

  Value *PtrSlot = nullptr;
  Type *PointerElemTy = nullptr;

public:
  PointerStorage(Value *PtrSlot, IRBuilderBase &IRB, Type *PointerElemTy)
      : VarStorage(IRB), PtrSlot(PtrSlot), PointerElemTy(PointerElemTy) {}
  std::unique_ptr<VarStorage> clone() const override {
    return std::make_unique<PointerStorage>(PtrSlot, IRB, PointerElemTy);
  }

  Value *getSlot() const override;
  // Load/store the pointee value through the pointer stored in PtrSlot.
  Value *loadValue() const override;
  void storeValue(Value *Val) override;
  // Load/store the pointer value itself from/to PtrSlot.
  Value *loadPointer() const;
  void storePointer(Value *Ptr);
  Type *getSlotType() const override;
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
  Value *getSlot() const override;
  Value *loadValue() const override;
  void storeValue(Value *Val) override;
  Type *getSlotType() const override;
  Type *getAllocatedType() const override;
  Type *getValueType() const override;
};

} // namespace proteus

#endif
