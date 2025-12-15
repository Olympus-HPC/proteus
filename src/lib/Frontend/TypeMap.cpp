#include "proteus/Frontend/TypeMap.hpp"

#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Type.h>

#include <cstddef>

namespace proteus {
using namespace llvm;

// Definitions for specializations.

llvm::Type *TypeMap<void>::get(LLVMContext &Ctx, std::size_t) {
  return Type::getVoidTy(Ctx);
}
llvm::Type *TypeMap<void>::getPointerElemType(LLVMContext &) { return nullptr; }
bool TypeMap<void>::isSigned() { return false; }

llvm::Type *TypeMap<float>::get(LLVMContext &Ctx, std::size_t) {
  return Type::getFloatTy(Ctx);
}
llvm::Type *TypeMap<float>::getPointerElemType(LLVMContext &) {
  return nullptr;
}
bool TypeMap<float>::isSigned() { return false; }

llvm::Type *TypeMap<float[]>::get(LLVMContext &Ctx, std::size_t NElem) {
  return ArrayType::get(Type::getFloatTy(Ctx), NElem);
}
llvm::Type *TypeMap<float[]>::getPointerElemType(LLVMContext &) {
  return nullptr;
}
bool TypeMap<float[]>::isSigned() { return false; }

llvm::Type *TypeMap<double>::get(LLVMContext &Ctx, std::size_t) {
  return Type::getDoubleTy(Ctx);
}
llvm::Type *TypeMap<double>::getPointerElemType(LLVMContext &) {
  return nullptr;
}
bool TypeMap<double>::isSigned() { return false; }

llvm::Type *TypeMap<double[]>::get(LLVMContext &Ctx, std::size_t NElem) {
  return ArrayType::get(Type::getDoubleTy(Ctx), NElem);
}
llvm::Type *TypeMap<double[]>::getPointerElemType(LLVMContext &) {
  return nullptr;
}
bool TypeMap<double[]>::isSigned() { return false; }

llvm::Type *TypeMap<size_t>::get(LLVMContext &Ctx, std::size_t) {
  return Type::getInt64Ty(Ctx);
}
llvm::Type *TypeMap<size_t>::getPointerElemType(LLVMContext &) {
  return nullptr;
}
bool TypeMap<size_t>::isSigned() { return false; }

llvm::Type *TypeMap<size_t[]>::get(LLVMContext &Ctx, std::size_t NElem) {
  return ArrayType::get(Type::getInt64Ty(Ctx), NElem);
}
llvm::Type *TypeMap<size_t[]>::getPointerElemType(LLVMContext &) {
  return nullptr;
}
bool TypeMap<size_t[]>::isSigned() { return false; }

llvm::Type *TypeMap<int>::get(LLVMContext &Ctx, std::size_t) {
  return Type::getInt32Ty(Ctx);
}
llvm::Type *TypeMap<int>::getPointerElemType(LLVMContext &) { return nullptr; }
bool TypeMap<int>::isSigned() { return true; }

llvm::Type *TypeMap<int[]>::get(LLVMContext &Ctx, std::size_t NElem) {
  return ArrayType::get(Type::getInt32Ty(Ctx), NElem);
}
llvm::Type *TypeMap<int[]>::getPointerElemType(LLVMContext &) {
  return nullptr;
}
bool TypeMap<int[]>::isSigned() { return true; }

llvm::Type *TypeMap<unsigned int>::get(LLVMContext &Ctx, std::size_t) {
  return Type::getInt32Ty(Ctx);
}
llvm::Type *TypeMap<unsigned int>::getPointerElemType(LLVMContext &) {
  return nullptr;
}
bool TypeMap<unsigned int>::isSigned() { return false; }

llvm::Type *TypeMap<unsigned int[]>::get(LLVMContext &Ctx, std::size_t NElem) {
  return ArrayType::get(Type::getInt32Ty(Ctx), NElem);
}
llvm::Type *TypeMap<unsigned int[]>::getPointerElemType(LLVMContext &) {
  return nullptr;
}
bool TypeMap<unsigned int[]>::isSigned() { return false; }

llvm::Type *TypeMap<int *>::get(LLVMContext &Ctx, std::size_t) {
  return PointerType::getUnqual(Ctx);
}
llvm::Type *TypeMap<int *>::getPointerElemType(LLVMContext &Ctx) {
  return Type::getInt32Ty(Ctx);
}
bool TypeMap<int *>::isSigned() { return true; }

llvm::Type *TypeMap<unsigned int *>::get(LLVMContext &Ctx, std::size_t) {
  return PointerType::getUnqual(Ctx);
}
llvm::Type *TypeMap<unsigned int *>::getPointerElemType(LLVMContext &Ctx) {
  return Type::getInt32Ty(Ctx);
}
bool TypeMap<unsigned int *>::isSigned() { return false; }

llvm::Type *TypeMap<bool>::get(LLVMContext &Ctx, std::size_t) {
  return Type::getInt1Ty(Ctx);
}
llvm::Type *TypeMap<bool>::getPointerElemType(LLVMContext &) { return nullptr; }
bool TypeMap<bool>::isSigned() { return false; }

llvm::Type *TypeMap<bool[]>::get(LLVMContext &Ctx, std::size_t NElem) {
  return ArrayType::get(Type::getInt1Ty(Ctx), NElem);
}
llvm::Type *TypeMap<bool[]>::getPointerElemType(LLVMContext &) {
  return nullptr;
}
bool TypeMap<bool[]>::isSigned() { return false; }

llvm::Type *TypeMap<double &>::get(LLVMContext &Ctx, std::size_t) {
  return PointerType::getUnqual(Ctx);
}
llvm::Type *TypeMap<double &>::getPointerElemType(LLVMContext &Ctx) {
  return Type::getDoubleTy(Ctx);
}
bool TypeMap<double &>::isSigned() { return false; }

llvm::Type *TypeMap<double *>::get(LLVMContext &Ctx, std::size_t) {
  return PointerType::getUnqual(Ctx);
}
llvm::Type *TypeMap<double *>::getPointerElemType(LLVMContext &Ctx) {
  return Type::getDoubleTy(Ctx);
}
bool TypeMap<double *>::isSigned() { return false; }

llvm::Type *TypeMap<float *>::get(LLVMContext &Ctx, std::size_t) {
  return PointerType::getUnqual(Ctx);
}
llvm::Type *TypeMap<float *>::getPointerElemType(LLVMContext &Ctx) {
  return Type::getFloatTy(Ctx);
}
bool TypeMap<float *>::isSigned() { return false; }

} // namespace proteus
