#ifndef PROTEUS_FRONTEND_TYPEMAP_H
#define PROTEUS_FRONTEND_TYPEMAP_H

#include "proteus/Frontend/IRType.h"

#include <cstddef>
#include <optional>

namespace proteus {

/// Maps a C++ type \c T to a backend-independent \c IRType descriptor.
///
/// This header has no dependency on any IR backend (LLVM, MLIR, …).
/// Backend-specific conversions live in dedicated headers such as
/// \c LLVMTypeMap.h.
template <typename T> struct TypeMap;

// Specialization declarations; definitions are in TypeMap.cpp.

template <> struct TypeMap<void> {
  static IRType get(std::size_t NElem = 0);
  static std::optional<IRType> getPointerElemType();
  static bool isSigned();
};

template <> struct TypeMap<float> {
  static IRType get(std::size_t NElem = 0);
  static std::optional<IRType> getPointerElemType();
  static bool isSigned();
};

template <> struct TypeMap<float[]> {
  static IRType get(std::size_t NElem);
  static std::optional<IRType> getPointerElemType();
  static bool isSigned();
};

template <> struct TypeMap<double> {
  static IRType get(std::size_t NElem = 0);
  static std::optional<IRType> getPointerElemType();
  static bool isSigned();
};

template <> struct TypeMap<double[]> {
  static IRType get(std::size_t NElem);
  static std::optional<IRType> getPointerElemType();
  static bool isSigned();
};

template <> struct TypeMap<size_t> {
  static IRType get(std::size_t NElem = 0);
  static std::optional<IRType> getPointerElemType();
  static bool isSigned();
};

template <> struct TypeMap<size_t[]> {
  static IRType get(std::size_t NElem);
  static std::optional<IRType> getPointerElemType();
  static bool isSigned();
};

template <> struct TypeMap<int> {
  static IRType get(std::size_t NElem = 0);
  static std::optional<IRType> getPointerElemType();
  static bool isSigned();
};

template <> struct TypeMap<int[]> {
  static IRType get(std::size_t NElem);
  static std::optional<IRType> getPointerElemType();
  static bool isSigned();
};

template <> struct TypeMap<unsigned int> {
  static IRType get(std::size_t NElem = 0);
  static std::optional<IRType> getPointerElemType();
  static bool isSigned();
};

template <> struct TypeMap<unsigned int[]> {
  static IRType get(std::size_t NElem);
  static std::optional<IRType> getPointerElemType();
  static bool isSigned();
};

template <> struct TypeMap<int *> {
  static IRType get(std::size_t NElem = 0);
  static std::optional<IRType> getPointerElemType();
  static bool isSigned();
};

template <> struct TypeMap<unsigned int *> {
  static IRType get(std::size_t NElem = 0);
  static std::optional<IRType> getPointerElemType();
  static bool isSigned();
};

template <> struct TypeMap<bool> {
  static IRType get(std::size_t NElem = 0);
  static std::optional<IRType> getPointerElemType();
  static bool isSigned();
};

template <> struct TypeMap<bool[]> {
  static IRType get(std::size_t NElem);
  static std::optional<IRType> getPointerElemType();
  static bool isSigned();
};

template <> struct TypeMap<double *> {
  static IRType get(std::size_t NElem = 0);
  static std::optional<IRType> getPointerElemType();
  static bool isSigned();
};

template <> struct TypeMap<float *> {
  static IRType get(std::size_t NElem = 0);
  static std::optional<IRType> getPointerElemType();
  static bool isSigned();
};

// Forward const types to their non-const equivalent.
template <typename T> struct TypeMap<const T> : TypeMap<T> {};

template <typename T> struct TypeMap<const T *> : TypeMap<T *> {};

// Forward reference types to their value types.
template <typename T> struct TypeMap<T &> : TypeMap<T> {};

} // namespace proteus

#endif // PROTEUS_FRONTEND_TYPEMAP_H
