#ifndef PROTEUS_FRONTEND_VAR_HPP
#define PROTEUS_FRONTEND_VAR_HPP

#include "proteus/Frontend/VarStorage.hpp"
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <type_traits>

namespace proteus {

class FuncBase;

using namespace llvm;

// Declare usual arithmetic conversion helper functions.
Value *convert(IRBuilderBase IRB, Value *V, Type *TargetType);
Type *getCommonType(const DataLayout &DL, Type *T1, Type *T2);

// Primary template declaration
template <typename T, typename = void> struct Var;
// Specialization for arithmetic types
template <typename T> struct Var<T, std::enable_if_t<std::is_arithmetic_v<T>>> {
  using ValueType = T;
  using ElemType = T;
  FuncBase &Fn;
  // Use opaque VarStorage to allow array/pointer
  // operator[] to return a Scalar Var that
  // points to the correct element.
  std::unique_ptr<VarStorage> Storage = nullptr;
  Var(std::unique_ptr<VarStorage> Storage, FuncBase &Fn)
      : Fn(Fn), Storage(std::move(Storage)) {}

  // // Conversion constructor
  // // TODO: Add an is_convertible check.
  template <typename U> Var(const Var<U> &V);

  Var(const Var &V) : Fn(V.Fn) { Storage = V.Storage->clone(); }

  Var(Var &&V) : Fn(V.Fn) { std::swap(Storage, V.Storage); }

  Var &operator=(Var &&V);

  // Assignment operators
  Var &operator=(const Var &V);

  template <typename U> Var &operator=(const Var<U> &V);

  template <typename U> Var &operator=(const U &ConstValue);

  // Arithmetic operators
  template <typename U>
  Var<std::common_type_t<T, U>> operator+(const Var<U> &Other) const;

  template <typename U, typename = std::enable_if_t<std::is_arithmetic_v<U>>>
  Var<std::common_type_t<T, U>> operator+(const U &ConstValue) const;

  template <typename U>
  Var<std::common_type_t<T, U>> operator-(const Var<U> &Other) const;

  template <typename U, typename = std::enable_if_t<std::is_arithmetic_v<U>>>
  Var<std::common_type_t<T, U>> operator-(const U &ConstValue) const;

  template <typename U>
  Var<std::common_type_t<T, U>> operator*(const Var<U> &Other) const;

  template <typename U, typename = std::enable_if_t<std::is_arithmetic_v<U>>>
  Var<std::common_type_t<T, U>> operator*(const U &ConstValue) const;

  template <typename U>
  Var<std::common_type_t<T, U>> operator/(const Var<U> &Other) const;

  template <typename U, typename = std::enable_if_t<std::is_arithmetic_v<U>>>
  Var<std::common_type_t<T, U>> operator/(const U &ConstValue) const;

  template <typename U>
  Var<std::common_type_t<T, U>> operator%(const Var<U> &Other) const;

  template <typename U, typename = std::enable_if_t<std::is_arithmetic_v<U>>>
  Var<std::common_type_t<T, U>> operator%(const U &ConstValue) const;

  // Compound assignment operators
  template <typename U> Var &operator+=(const Var<U> &Other);

  template <typename U> Var &operator+=(const U &ConstValue);

  template <typename U> Var &operator-=(const Var<U> &Other);

  template <typename U> Var &operator-=(const U &ConstValue);

  template <typename U> Var &operator*=(const Var<U> &Other);

  template <typename U> Var &operator*=(const U &ConstValue);

  template <typename U> Var &operator/=(const Var<U> &Other);

  template <typename U> Var &operator/=(const U &ConstValue);

  template <typename U> Var &operator%=(const Var<U> &Other);

  template <typename U> Var &operator%=(const U &ConstValue);

  // Comparison operators
  template <typename U>
  std::enable_if_t<std::is_arithmetic_v<U>, Var<bool>>
  operator>(const Var<U> &Other) const;

  template <typename U>
  std::enable_if_t<std::is_arithmetic_v<U>, Var<bool>>
  operator>=(const Var<U> &Other) const;

  template <typename U>
  std::enable_if_t<std::is_arithmetic_v<U>, Var<bool>>
  operator<(const Var<U> &Other) const;

  template <typename U>
  std::enable_if_t<std::is_arithmetic_v<U>, Var<bool>>
  operator<=(const Var<U> &Other) const;

  template <typename U>
  std::enable_if_t<std::is_arithmetic_v<U>, Var<bool>>
  operator==(const Var<U> &Other) const;

  template <typename U>
  std::enable_if_t<std::is_arithmetic_v<U>, Var<bool>>
  operator!=(const Var<U> &Other) const;

  template <typename U>
  std::enable_if_t<std::is_arithmetic_v<U>, Var<bool>>
  operator>(const U &ConstValue) const;

  template <typename U>
  std::enable_if_t<std::is_arithmetic_v<U>, Var<bool>>
  operator>=(const U &ConstValue) const;

  template <typename U>
  std::enable_if_t<std::is_arithmetic_v<U>, Var<bool>>
  operator<(const U &ConstValue) const;

  template <typename U>
  std::enable_if_t<std::is_arithmetic_v<U>, Var<bool>>
  operator<=(const U &ConstValue) const;

  template <typename U>
  std::enable_if_t<std::is_arithmetic_v<U>, Var<bool>>
  operator==(const U &ConstValue) const;

  template <typename U>
  std::enable_if_t<std::is_arithmetic_v<U>, Var<bool>>
  operator!=(const U &ConstValue) const;
};

// Specialization for array types
template <typename T> struct Var<T, std::enable_if_t<std::is_array_v<T>>> {
  using ValueType = T;
  using ElemType = std::remove_extent_t<T>;
  FuncBase &Fn;
  std::unique_ptr<ArrayStorage> Storage = nullptr;

  Var(std::unique_ptr<ArrayStorage> Storage, FuncBase &Fn)
      : Fn(Fn), Storage(std::move(Storage)) {}

  Var<ElemType> operator[](size_t Index);

  template <typename IdxT>
  std::enable_if_t<std::is_integral_v<IdxT>, Var<ElemType>>
  operator[](const Var<IdxT> &Index);
};

// Specialization for pointer types
template <typename T> struct Var<T, std::enable_if_t<std::is_pointer_v<T>>> {
  using ValueType = T;
  using ElemType = std::remove_pointer_t<T>;
  FuncBase &Fn;
  std::unique_ptr<PointerStorage> Storage = nullptr;

  Var(std::unique_ptr<PointerStorage> Storage, FuncBase &Fn)
      : Fn(Fn), Storage(std::move(Storage)) {}

  Var<ElemType> operator[](size_t Index);

  template <typename IdxT>
  std::enable_if_t<std::is_arithmetic_v<IdxT>, Var<ElemType>>
  operator[](const Var<IdxT> &Index);

  Var<ElemType> operator*();

  template <typename OffsetT>
  std::enable_if_t<std::is_arithmetic_v<OffsetT>,
                   Var<T, std::enable_if_t<std::is_pointer_v<T>>>>
  operator+(const Var<OffsetT> &Offset) const;

  template <typename OffsetT>
  std::enable_if_t<std::is_arithmetic_v<OffsetT>,
                   Var<T, std::enable_if_t<std::is_pointer_v<T>>>>
  operator+(OffsetT Offset) const;

  template <typename OffsetT>
  friend std::enable_if_t<std::is_arithmetic_v<OffsetT>,
                          Var<T, std::enable_if_t<std::is_pointer_v<T>>>>
  operator+(OffsetT Offset, const Var &Ptr) {
    return Ptr + Offset;
  }
};

// Non-member arithmetic operators for Var
template <typename T, typename U>
std::enable_if_t<std::is_arithmetic_v<T> && std::is_arithmetic_v<U>,
                 Var<std::common_type_t<T, U>>>
operator+(const T &ConstValue, const Var<U> &Var);

template <typename T, typename U>
std::enable_if_t<std::is_arithmetic_v<T> && std::is_arithmetic_v<U>,
                 Var<std::common_type_t<T, U>>>
operator-(const T &ConstValue, const Var<U> &V);

template <typename T, typename U>
std::enable_if_t<std::is_arithmetic_v<T> && std::is_arithmetic_v<U>,
                 Var<std::common_type_t<T, U>>>
operator*(const T &ConstValue, const Var<U> &V);

template <typename T, typename U>
std::enable_if_t<std::is_arithmetic_v<T> && std::is_arithmetic_v<U>,
                 Var<std::common_type_t<T, U>>>
operator/(const T &ConstValue, const Var<U> &V);

template <typename T, typename U>
std::enable_if_t<std::is_arithmetic_v<T> && std::is_arithmetic_v<U>,
                 Var<std::common_type_t<T, U>>>
operator%(const T &ConstValue, const Var<U> &V);

// Math intrinsics for Var
template <typename T>
std::enable_if_t<std::is_same_v<T, float>, Var<T>> powf(const Var<T> &L,
                                                        const Var<T> &R);

template <typename T>
std::enable_if_t<std::is_same_v<T, float>, Var<T>> sqrtf(const Var<T> &R);

template <typename T>
std::enable_if_t<std::is_arithmetic_v<T>, Var<T>> min(const Var<T> &L,
                                                      const Var<T> &R);
} // namespace proteus

#endif // PROTEUS_FRONTEND_VAR_HPP
