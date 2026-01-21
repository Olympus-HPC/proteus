#ifndef PROTEUS_FRONTEND_TYPEMAP_H
#define PROTEUS_FRONTEND_TYPEMAP_H

#include <cstddef>

namespace llvm {
class LLVMContext;
class Type;
} // namespace llvm

namespace proteus {

template <typename T> struct TypeMap;

// Specialization declarations, definitions are in TypeMap.cpp.
template <> struct TypeMap<void> {
  static llvm::Type *get(llvm::LLVMContext &Ctx, std::size_t NElem = 0);
  static llvm::Type *getPointerElemType(llvm::LLVMContext &Ctx);
  static bool isSigned();
};

template <> struct TypeMap<float> {
  static llvm::Type *get(llvm::LLVMContext &Ctx, std::size_t NElem = 0);
  static llvm::Type *getPointerElemType(llvm::LLVMContext &Ctx);
  static bool isSigned();
};

template <> struct TypeMap<float[]> {
  static llvm::Type *get(llvm::LLVMContext &Ctx, std::size_t NElem);
  static llvm::Type *getPointerElemType(llvm::LLVMContext &Ctx);
  static bool isSigned();
};

template <> struct TypeMap<double> {
  static llvm::Type *get(llvm::LLVMContext &Ctx, std::size_t NElem = 0);
  static llvm::Type *getPointerElemType(llvm::LLVMContext &Ctx);
  static bool isSigned();
};

template <> struct TypeMap<double[]> {
  static llvm::Type *get(llvm::LLVMContext &Ctx, std::size_t NElem);
  static llvm::Type *getPointerElemType(llvm::LLVMContext &Ctx);
  static bool isSigned();
};

template <> struct TypeMap<size_t> {
  static llvm::Type *get(llvm::LLVMContext &Ctx, std::size_t NElem = 0);
  static llvm::Type *getPointerElemType(llvm::LLVMContext &Ctx);
  static bool isSigned();
};

template <> struct TypeMap<size_t[]> {
  static llvm::Type *get(llvm::LLVMContext &Ctx, std::size_t NElem);
  static llvm::Type *getPointerElemType(llvm::LLVMContext &Ctx);
  static bool isSigned();
};

template <> struct TypeMap<int> {
  static llvm::Type *get(llvm::LLVMContext &Ctx, std::size_t NElem = 0);
  static llvm::Type *getPointerElemType(llvm::LLVMContext &Ctx);
  static bool isSigned();
};

template <> struct TypeMap<int[]> {
  static llvm::Type *get(llvm::LLVMContext &Ctx, std::size_t NElem);
  static llvm::Type *getPointerElemType(llvm::LLVMContext &Ctx);
  static bool isSigned();
};

template <> struct TypeMap<unsigned int> {
  static llvm::Type *get(llvm::LLVMContext &Ctx, std::size_t NElem = 0);
  static llvm::Type *getPointerElemType(llvm::LLVMContext &Ctx);
  static bool isSigned();
};

template <> struct TypeMap<unsigned int[]> {
  static llvm::Type *get(llvm::LLVMContext &Ctx, std::size_t NElem);
  static llvm::Type *getPointerElemType(llvm::LLVMContext &Ctx);
  static bool isSigned();
};

template <> struct TypeMap<int *> {
  static llvm::Type *get(llvm::LLVMContext &Ctx, std::size_t NElem = 0);
  static llvm::Type *getPointerElemType(llvm::LLVMContext &Ctx);
  static bool isSigned();
};

template <> struct TypeMap<unsigned int *> {
  static llvm::Type *get(llvm::LLVMContext &Ctx, std::size_t NElem = 0);
  static llvm::Type *getPointerElemType(llvm::LLVMContext &Ctx);
  static bool isSigned();
};

template <> struct TypeMap<bool> {
  static llvm::Type *get(llvm::LLVMContext &Ctx, std::size_t NElem = 0);
  static llvm::Type *getPointerElemType(llvm::LLVMContext &Ctx);
  static bool isSigned();
};

template <> struct TypeMap<bool[]> {
  static llvm::Type *get(llvm::LLVMContext &Ctx, std::size_t NElem);
  static llvm::Type *getPointerElemType(llvm::LLVMContext &Ctx);
  static bool isSigned();
};

template <> struct TypeMap<double *> {
  static llvm::Type *get(llvm::LLVMContext &Ctx, std::size_t NElem = 0);
  static llvm::Type *getPointerElemType(llvm::LLVMContext &Ctx);
  static bool isSigned();
};

template <> struct TypeMap<float *> {
  static llvm::Type *get(llvm::LLVMContext &Ctx, std::size_t NElem = 0);
  static llvm::Type *getPointerElemType(llvm::LLVMContext &Ctx);
  static bool isSigned();
};

// Forward const types to their non-const equivalent.
template <typename T> struct TypeMap<const T> : TypeMap<T> {};

template <typename T> struct TypeMap<const T *> : TypeMap<T *> {};

// Forward reference types to their value types.
template <typename T> struct TypeMap<T &> : TypeMap<T> {};

} // namespace proteus

#endif // PROTEUS_FRONTEND_TYPEMAP_H
