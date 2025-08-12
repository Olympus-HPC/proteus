#ifndef PROTEUS_FRONTEND_TYPEMAP_HPP
#define PROTEUS_FRONTEND_TYPEMAP_HPP

#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Type.h>

#include <cstddef>

namespace proteus {
using namespace llvm;

template <typename T> struct TypeMap;

template <> struct TypeMap<void> {
  static Type *get(llvm::LLVMContext &Ctx) { return Type::getVoidTy(Ctx); }

  static Type *getPointerElemType(llvm::LLVMContext &) { return nullptr; }
};

template <> struct TypeMap<float> {
  static Type *get(llvm::LLVMContext &Ctx) { return Type::getFloatTy(Ctx); }

  static Type *getPointerElemType(llvm::LLVMContext &) { return nullptr; }
};

template <> struct TypeMap<double> {
  static Type *get(llvm::LLVMContext &Ctx) { return Type::getDoubleTy(Ctx); }

  static Type *getPointerElemType(llvm::LLVMContext &) { return nullptr; }
};

template <> struct TypeMap<size_t> {
  static Type *get(llvm::LLVMContext &Ctx) { return Type::getInt64Ty(Ctx); }

  static Type *getPointerElemType(llvm::LLVMContext &) { return nullptr; }
};

template <> struct TypeMap<int> {
  static Type *get(llvm::LLVMContext &Ctx) { return Type::getInt32Ty(Ctx); }

  static Type *getPointerElemType(llvm::LLVMContext &) { return nullptr; }

  static bool isSigned() { return true; }
};

template <> struct TypeMap<unsigned int> {
  static Type *get(llvm::LLVMContext &Ctx) { return Type::getInt32Ty(Ctx); }

  static Type *getPointerElemType(llvm::LLVMContext &) { return nullptr; }

  static bool isSigned() { return false; }
};

template <> struct TypeMap<int *> {
  static Type *get(llvm::LLVMContext &Ctx) {
    return PointerType::getUnqual(Ctx);
  }

  static Type *getPointerElemType(llvm::LLVMContext &Ctx) {
    return Type::getInt32Ty(Ctx);
  }
};

template <> struct TypeMap<unsigned int *> {
  static Type *get(llvm::LLVMContext &Ctx) {
    return PointerType::getUnqual(Ctx);
  }

  static Type *getPointerElemType(llvm::LLVMContext &Ctx) {
    return Type::getInt32Ty(Ctx);
  }
};

template <> struct TypeMap<bool> {
  static Type *get(llvm::LLVMContext &Ctx) { return Type::getInt1Ty(Ctx); }

  static Type *getPointerElemType(llvm::LLVMContext &) { return nullptr; }
};

template <> struct TypeMap<double &> {
  static Type *get(llvm::LLVMContext &Ctx) {
    return PointerType::getUnqual(Ctx);
  }

  static Type *getPointerElemType(llvm::LLVMContext &Ctx) {
    return Type::getDoubleTy(Ctx);
  }
};

template <> struct TypeMap<double *> {
  static Type *get(llvm::LLVMContext &Ctx) {
    return PointerType::getUnqual(Ctx);
  }

  static Type *getPointerElemType(llvm::LLVMContext &Ctx) {
    return Type::getDoubleTy(Ctx);
  }
};

template <> struct TypeMap<float *> {
  static Type *get(llvm::LLVMContext &Ctx) {
    return PointerType::getUnqual(Ctx);
  }

  static Type *getPointerElemType(llvm::LLVMContext &Ctx) {
    return Type::getFloatTy(Ctx);
  }
};

} // namespace proteus

#endif // PROTEUS_FRONTEND_TYPEMAP_HPP
