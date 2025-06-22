#ifndef PROTEUS_PASS_TYPES_H
#define PROTEUS_PASS_TYPES_H

#include <llvm/IR/Module.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Type.h>

#include "Helpers.h"

namespace proteus {

using namespace llvm;

struct ProteusTypes {
  ProteusTypes(Module &M)
      : PtrTy(PointerType::getUnqual(M.getContext())),
        VoidTy(Type::getVoidTy(M.getContext())),
        Int1Ty(Type::getInt1Ty(M.getContext())),
        Int8Ty(Type::getInt8Ty(M.getContext())),
        Int32Ty(Type::getInt32Ty(M.getContext())),
        Int64Ty(Type::getInt64Ty(M.getContext())),
        Int128Ty(Type::getInt128Ty(M.getContext())) {
    // llvm.global.annotations entry format:
    //  ptr: (addrspace 1) Function pointer
    //  ptr: (addrspace 4) Annotations string
    //  ptr: (addrspace 4) Source file string
    //  i32: Line number,
    //  ptr: (addrspace 1) Arguments pointer
    if (isDeviceCompilation(M)) {
      constexpr unsigned GlobalAddressSpace = 1;
      constexpr unsigned ConstAddressSpace = 4;
      GlobalAnnotationEltTy = StructType::get(
          PointerType::get(M.getContext(), GlobalAddressSpace),
          PointerType::get(M.getContext(), ConstAddressSpace),
          PointerType::get(M.getContext(), ConstAddressSpace), Int32Ty,
          PointerType::get(M.getContext(), GlobalAddressSpace));
    } else
      GlobalAnnotationEltTy =
          StructType::get(PtrTy, PtrTy, PtrTy, Int32Ty, PtrTy);
  }

  Type *PtrTy = nullptr;
  Type *VoidTy = nullptr;
  Type *Int1Ty = nullptr;
  Type *Int8Ty = nullptr;
  Type *Int32Ty = nullptr;
  Type *Int64Ty = nullptr;
  Type *Int128Ty = nullptr;
  StructType *GlobalAnnotationEltTy = nullptr;
};

} // namespace proteus

#endif
