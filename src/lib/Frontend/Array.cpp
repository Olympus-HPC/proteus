#include "proteus/Error.h"
#include "proteus/Frontend/Array.hpp"
#include "proteus/Frontend/Func.hpp"

namespace proteus {

Array::Array(Value *BasePointer, FuncBase &Fn, Type *ArrayType, AddressSpace AT)
    : BasePointer(BasePointer), Fn(Fn), ArrayType(ArrayType), AT(AT) {}

Var Array::operator[](size_t Index) {
  auto &IRB = Fn.getIRBuilder();
  auto *GEP = IRB.CreateConstInBoundsGEP2_64(ArrayType, BasePointer, 0, Index);
  Type *ElemTy = ArrayType->getArrayElementType();
  auto *BasePtrTy = cast<PointerType>(BasePointer->getType());
  unsigned AddrSpace = BasePtrTy->getAddressSpace();
  Type *ElemPtrTy = PointerType::get(ElemTy, AddrSpace);
  auto &ResultVar = Fn.declVarInternal("res.", ElemPtrTy, ElemTy);
  ResultVar.storePointer(GEP);
  return ResultVar;
}

Var Array::operator[](const Var &Index) {
  auto &IRB = Fn.getIRBuilder();
  Value *Idx = Index.getValue();
  if (!Idx->getType()->isIntegerTy())
    PROTEUS_FATAL_ERROR("Expected integer index for array GEP");
  Value *Zero = llvm::ConstantInt::get(Idx->getType(), 0);
  auto *GEP = IRB.CreateInBoundsGEP(ArrayType, BasePointer, {Zero, Idx});
  Type *ElemTy = ArrayType->getArrayElementType();
  auto *BasePtrTy = cast<PointerType>(BasePointer->getType());
  unsigned AddrSpace = BasePtrTy->getAddressSpace();
  Type *ElemPtrTy = PointerType::get(ElemTy, AddrSpace);
  auto &ResultVar = Fn.declVarInternal("res.", ElemPtrTy, ElemTy);
  ResultVar.storePointer(GEP);
  return ResultVar;
}
} // namespace proteus
