#ifndef PROTEUS_FRONTEND_VARSTORAGE_H
#define PROTEUS_FRONTEND_VARSTORAGE_H

#include "proteus/Frontend/IRType.h"
#include "proteus/Frontend/IRValue.h"

#include <memory>

namespace llvm {
class IRBuilderBase;
} // namespace llvm

namespace proteus {

class VarStorage {

protected:
  llvm::IRBuilderBase &IRB;

public:
  VarStorage(llvm::IRBuilderBase &IRB) : IRB(IRB) {}
  virtual ~VarStorage() = default;

  virtual IRValue getSlot() const = 0;
  // Load/store the logical value represented by this storage.
  virtual IRValue loadValue() const = 0;
  virtual void storeValue(IRValue Val) = 0;
  virtual IRType getAllocatedType() const = 0;
  virtual IRType getValueType() const = 0;
  virtual std::unique_ptr<VarStorage> clone() const = 0;
};

class ScalarStorage : public VarStorage {
  IRValue Slot;
  IRType IRValTy;

public:
  ScalarStorage(IRValue Slot, llvm::IRBuilderBase &IRB, IRType IRValTy)
      : VarStorage(IRB), Slot(Slot), IRValTy(IRValTy) {}
  std::unique_ptr<VarStorage> clone() const override {
    return std::make_unique<ScalarStorage>(Slot, IRB, IRValTy);
  }

  IRValue getSlot() const override;
  IRValue loadValue() const override;
  void storeValue(IRValue Val) override;
  IRType getAllocatedType() const override;
  IRType getValueType() const override;
};

class PointerStorage : public VarStorage {

  IRValue PtrSlot;
  IRType ElemIRTy;

public:
  PointerStorage(IRValue PtrSlot, llvm::IRBuilderBase &IRB, IRType ElemIRTy)
      : VarStorage(IRB), PtrSlot(PtrSlot), ElemIRTy(ElemIRTy) {}
  std::unique_ptr<VarStorage> clone() const override {
    return std::make_unique<PointerStorage>(PtrSlot, IRB, ElemIRTy);
  }

  IRValue getSlot() const override;
  // Load/store the pointee value through the pointer stored in PtrSlot.
  IRValue loadValue() const override;
  void storeValue(IRValue Val) override;
  // Load/store the pointer value itself from/to PtrSlot.
  IRValue loadPointer() const;
  void storePointer(IRValue Ptr);
  IRType getAllocatedType() const override;
  IRType getValueType() const override;
};

class ArrayStorage : public VarStorage {

  IRValue BasePointer;
  std::size_t NElem = 0;
  IRType ElemIRTy;

public:
  ArrayStorage(IRValue BasePointer, llvm::IRBuilderBase &IRB, std::size_t NElem,
               IRType ElemIRTy)
      : VarStorage(IRB), BasePointer(BasePointer), NElem(NElem),
        ElemIRTy(ElemIRTy) {}
  std::unique_ptr<VarStorage> clone() const override {
    return std::make_unique<ArrayStorage>(BasePointer, IRB, NElem, ElemIRTy);
  }
  IRValue getSlot() const override;
  IRValue loadValue() const override;
  void storeValue(IRValue Val) override;
  IRType getAllocatedType() const override;
  IRType getValueType() const override;
};

} // namespace proteus

#endif
