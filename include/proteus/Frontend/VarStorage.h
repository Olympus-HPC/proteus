#ifndef PROTEUS_FRONTEND_VARSTORAGE_H
#define PROTEUS_FRONTEND_VARSTORAGE_H

#include <memory>

namespace llvm {
class Value;
class IRBuilderBase;
class Type;
class ArrayType;
} // namespace llvm

namespace proteus {

class VarStorage {

protected:
  llvm::IRBuilderBase &IRB;

public:
  VarStorage(llvm::IRBuilderBase &IRB) : IRB(IRB) {}
  virtual ~VarStorage() = default;

  virtual llvm::Value *getSlot() const = 0;
  // Load/store the logical value represented by this storage.
  virtual llvm::Value *loadValue() const = 0;
  virtual void storeValue(llvm::Value *Val) = 0;
  virtual llvm::Type *getSlotType() const = 0;
  virtual llvm::Type *getAllocatedType() const = 0;
  virtual llvm::Type *getValueType() const = 0;
  virtual std::unique_ptr<VarStorage> clone() const = 0;
};

class ScalarStorage : public VarStorage {
  llvm::Value *Slot = nullptr;

public:
  ScalarStorage(llvm::Value *Slot, llvm::IRBuilderBase &IRB)
      : VarStorage(IRB), Slot(Slot) {}
  std::unique_ptr<VarStorage> clone() const override {
    return std::make_unique<ScalarStorage>(Slot, IRB);
  }

  llvm::Value *getSlot() const override;
  llvm::Value *loadValue() const override;
  void storeValue(llvm::Value *Val) override;
  llvm::Type *getSlotType() const override;
  llvm::Type *getAllocatedType() const override;
  llvm::Type *getValueType() const override;
};

class PointerStorage : public VarStorage {

  llvm::Value *PtrSlot = nullptr;
  llvm::Type *PointerElemTy = nullptr;

public:
  PointerStorage(llvm::Value *PtrSlot, llvm::IRBuilderBase &IRB,
                 llvm::Type *PointerElemTy)
      : VarStorage(IRB), PtrSlot(PtrSlot), PointerElemTy(PointerElemTy) {}
  std::unique_ptr<VarStorage> clone() const override {
    return std::make_unique<PointerStorage>(PtrSlot, IRB, PointerElemTy);
  }

  llvm::Value *getSlot() const override;
  // Load/store the pointee value through the pointer stored in PtrSlot.
  llvm::Value *loadValue() const override;
  void storeValue(llvm::Value *Val) override;
  // Load/store the pointer value itself from/to PtrSlot.
  llvm::Value *loadPointer() const;
  void storePointer(llvm::Value *Ptr);
  llvm::Type *getSlotType() const override;
  llvm::Type *getAllocatedType() const override;
  llvm::Type *getValueType() const override;
};

class ArrayStorage : public VarStorage {

  llvm::Value *BasePointer = nullptr;
  llvm::ArrayType *ArrayTy = nullptr;

public:
  ArrayStorage(llvm::Value *BasePointer, llvm::IRBuilderBase &IRB,
               llvm::ArrayType *ArrayTy)
      : VarStorage(IRB), BasePointer(BasePointer), ArrayTy(ArrayTy) {}
  std::unique_ptr<VarStorage> clone() const override {
    return std::make_unique<ArrayStorage>(BasePointer, IRB, ArrayTy);
  }
  llvm::Value *getSlot() const override;
  llvm::Value *loadValue() const override;
  void storeValue(llvm::Value *Val) override;
  llvm::Type *getSlotType() const override;
  llvm::Type *getAllocatedType() const override;
  llvm::Type *getValueType() const override;
};

} // namespace proteus

#endif
