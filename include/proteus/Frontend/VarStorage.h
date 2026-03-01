#ifndef PROTEUS_FRONTEND_VARSTORAGE_H
#define PROTEUS_FRONTEND_VARSTORAGE_H

#include "proteus/Frontend/IRType.h"

#include <memory>

namespace llvm {
class Value;
class IRBuilderBase;
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
  virtual IRType getAllocatedType() const = 0;
  virtual IRType getValueType() const = 0;
  virtual std::unique_ptr<VarStorage> clone() const = 0;
};

class ScalarStorage : public VarStorage {
  llvm::Value *Slot = nullptr;
  IRType IRValTy;

public:
  ScalarStorage(llvm::Value *Slot, llvm::IRBuilderBase &IRB, IRType IRValTy)
      : VarStorage(IRB), Slot(Slot), IRValTy(IRValTy) {}
  std::unique_ptr<VarStorage> clone() const override {
    return std::make_unique<ScalarStorage>(Slot, IRB, IRValTy);
  }

  llvm::Value *getSlot() const override;
  llvm::Value *loadValue() const override;
  void storeValue(llvm::Value *Val) override;
  IRType getAllocatedType() const override;
  IRType getValueType() const override;
};

class PointerStorage : public VarStorage {

  llvm::Value *PtrSlot = nullptr;
  IRType ElemIRTy;

public:
  PointerStorage(llvm::Value *PtrSlot, llvm::IRBuilderBase &IRB,
                 IRType ElemIRTy)
      : VarStorage(IRB), PtrSlot(PtrSlot), ElemIRTy(ElemIRTy) {}
  std::unique_ptr<VarStorage> clone() const override {
    return std::make_unique<PointerStorage>(PtrSlot, IRB, ElemIRTy);
  }

  llvm::Value *getSlot() const override;
  // Load/store the pointee value through the pointer stored in PtrSlot.
  llvm::Value *loadValue() const override;
  void storeValue(llvm::Value *Val) override;
  // Load/store the pointer value itself from/to PtrSlot.
  llvm::Value *loadPointer() const;
  void storePointer(llvm::Value *Ptr);
  IRType getAllocatedType() const override;
  IRType getValueType() const override;
};

class ArrayStorage : public VarStorage {

  llvm::Value *BasePointer = nullptr;
  std::size_t NElem = 0;
  IRType ElemIRTy;

public:
  ArrayStorage(llvm::Value *BasePointer, llvm::IRBuilderBase &IRB,
               std::size_t NElem, IRType ElemIRTy)
      : VarStorage(IRB), BasePointer(BasePointer), NElem(NElem),
        ElemIRTy(ElemIRTy) {}
  std::unique_ptr<VarStorage> clone() const override {
    return std::make_unique<ArrayStorage>(BasePointer, IRB, NElem, ElemIRTy);
  }
  llvm::Value *getSlot() const override;
  llvm::Value *loadValue() const override;
  void storeValue(llvm::Value *Val) override;
  IRType getAllocatedType() const override;
  IRType getValueType() const override;
};

} // namespace proteus

#endif
