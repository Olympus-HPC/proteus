#ifndef PROTEUS_FRONTEND_VAR_HPP
#define PROTEUS_FRONTEND_VAR_HPP

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <type_traits>
#include "proteus/Frontend/VarStorage.hpp"

namespace proteus {

class FuncBase;

using namespace llvm;

// Declare usual arithmetic conversion helper functions.
Value *convert(IRBuilderBase IRB, Value *V, Type *TargetType);
Type *getCommonType(const DataLayout &DL, Type *T1, Type *T2);

// Primary template declaration
template <typename T, typename = void>
struct VarTT;
// Specialization for arithmetic types
template <typename T>
struct VarTT<T, std::enable_if_t<std::is_arithmetic_v<T>>> {
  using ValueType = T;
  using ElemType = T;
  FuncBase &Fn;
  // Use opaque VarStorage to allow array/pointer
  // operator[] to return a Scalar Var that
  // points to the correct element.
  std::unique_ptr<VarStorage> Storage = nullptr;
  VarTT(std::unique_ptr<VarStorage> Storage, FuncBase &Fn)
    : Fn(Fn), Storage(std::move(Storage)) {
  }

  // // Conversion constructor
  // // TODO: Add an is_convertible check.
  template<typename U>
  VarTT(const VarTT<U> &Var);

  VarTT(const VarTT &Var) 
  : Fn(Var.Fn) { 
    Storage = Var.Storage->clone();
  }

  
  VarTT(VarTT &&Var) 
  : Fn(Var.Fn) {
    std::swap(Storage, Var.Storage);
  }

  VarTT &operator=(VarTT &&Var);
  
  // Assignment operators
  VarTT &operator=(const VarTT &Var);
  
  template<typename U>
  VarTT &operator=(const VarTT<U> &Var);
  
  template <typename U>
  VarTT &operator=(const U &ConstValue);
  
  // Arithmetic operators
  template <typename U>
  VarTT<std::common_type_t<T, U>> operator+(const VarTT<U> &Other) const;
  
  template <typename U, typename = std::enable_if_t<std::is_arithmetic_v<U>>>
  VarTT<std::common_type_t<T, U>> operator+(const U &ConstValue) const;
  
  template <typename U>
  VarTT<std::common_type_t<T, U>> operator-(const VarTT<U> &Other) const;
  
  template <typename U, typename = std::enable_if_t<std::is_arithmetic_v<U>>>
  VarTT<std::common_type_t<T, U>> operator-(const U &ConstValue) const;
  
  template <typename U>
  VarTT<std::common_type_t<T, U>> operator*(const VarTT<U> &Other) const;
  
  template <typename U, typename = std::enable_if_t<std::is_arithmetic_v<U>>>
  VarTT<std::common_type_t<T, U>> operator*(const U &ConstValue) const;
  
  template <typename U>
  VarTT<std::common_type_t<T, U>> operator/(const VarTT<U> &Other) const;
  
  template <typename U, typename = std::enable_if_t<std::is_arithmetic_v<U>>>
  VarTT<std::common_type_t<T, U>> operator/(const U &ConstValue) const;
  
  template <typename U>
  VarTT<std::common_type_t<T, U>> operator%(const VarTT<U> &Other) const;
  
  template <typename U, typename = std::enable_if_t<std::is_arithmetic_v<U>>>
  VarTT<std::common_type_t<T, U>> operator%(const U &ConstValue) const;
  
  // Compound assignment operators
  template <typename U>
  VarTT &operator+=(const VarTT<U> &Other);
  
  template <typename U>
  VarTT &operator+=(const U &ConstValue);
  
  template <typename U>
  VarTT &operator-=(const VarTT<U> &Other);
  
  template <typename U>
  VarTT &operator-=(const U &ConstValue);
  
  template <typename U>
  VarTT &operator*=(const VarTT<U> &Other);
  
  template <typename U>
  VarTT &operator*=(const U &ConstValue);
  
  template <typename U>
  VarTT &operator/=(const VarTT<U> &Other);
  
  template <typename U>
  VarTT &operator/=(const U &ConstValue);
  
  template <typename U>
  VarTT &operator%=(const VarTT<U> &Other);
  
  template <typename U>
  VarTT &operator%=(const U &ConstValue);
  
  // Comparison operators
  template <typename U>
  std::enable_if_t<std::is_arithmetic_v<U>, VarTT<bool>>
  operator>(const VarTT<U> &Other) const;
  
  template <typename U>
  std::enable_if_t<std::is_arithmetic_v<U>, VarTT<bool>>
  operator>=(const VarTT<U> &Other) const;
  
  template <typename U>
  std::enable_if_t<std::is_arithmetic_v<U>, VarTT<bool>>
  operator<(const VarTT<U> &Other) const;
  
  template <typename U>
  std::enable_if_t<std::is_arithmetic_v<U>, VarTT<bool>>
  operator<=(const VarTT<U> &Other) const;
  
  template <typename U>
  std::enable_if_t<std::is_arithmetic_v<U>, VarTT<bool>>
  operator==(const VarTT<U> &Other) const;
  
  template <typename U>
  std::enable_if_t<std::is_arithmetic_v<U>, VarTT<bool>>
  operator!=(const VarTT<U> &Other) const;
  
  template <typename U>
  std::enable_if_t<std::is_arithmetic_v<U>, VarTT<bool>>
  operator>(const U &ConstValue) const;
  
  template <typename U>
  std::enable_if_t<std::is_arithmetic_v<U>, VarTT<bool>>
  operator>=(const U &ConstValue) const;
  
  template <typename U>
  std::enable_if_t<std::is_arithmetic_v<U>, VarTT<bool>>
  operator<(const U &ConstValue) const;
  
  template <typename U>
  std::enable_if_t<std::is_arithmetic_v<U>, VarTT<bool>>
  operator<=(const U &ConstValue) const;
  
  template <typename U>
  std::enable_if_t<std::is_arithmetic_v<U>, VarTT<bool>>
  operator==(const U &ConstValue) const;
  
  template <typename U>
  std::enable_if_t<std::is_arithmetic_v<U>, VarTT<bool>>
  operator!=(const U &ConstValue) const;
};

// Specialization for array types
template <typename T>
struct VarTT<T, std::enable_if_t<std::is_array_v<T>>> {
  using ValueType = T;
  using ElemType = std::remove_extent_t<T>;
  FuncBase &Fn;
  std::unique_ptr<ArrayStorage> Storage = nullptr;

  VarTT(std::unique_ptr<ArrayStorage> Storage, FuncBase &Fn)
    : Fn(Fn), Storage(std::move(Storage)) {}
  
  VarTT<ElemType> operator[](size_t Index);

  template <typename IdxT>
  std::enable_if_t<std::is_integral_v<IdxT>, VarTT<ElemType> >
  operator[](const VarTT<IdxT> &Index);
};

// Specialization for pointer types
template <typename T>
struct VarTT<T, std::enable_if_t<std::is_pointer_v<T>>> {
  using ValueType = T;
  using ElemType = std::remove_pointer_t<T>;
  FuncBase &Fn;
  std::unique_ptr<PointerStorage> Storage = nullptr;

  VarTT(std::unique_ptr<PointerStorage> Storage, FuncBase &Fn)
    : Fn(Fn), Storage(std::move(Storage)) {}
  
  VarTT<ElemType> operator[](size_t Index);

  template <typename IdxT>
  std::enable_if_t<std::is_arithmetic_v<IdxT>, VarTT<ElemType>>
  operator[](const VarTT<IdxT> &Index);

  VarTT<ElemType> operator*();

  template <typename OffsetT>
  std::enable_if_t<std::is_arithmetic_v<OffsetT>, VarTT<T, std::enable_if_t<std::is_pointer_v<T>>>>
  operator+(const VarTT<OffsetT> &Offset) const;

  template <typename OffsetT>
  std::enable_if_t<std::is_arithmetic_v<OffsetT>, VarTT<T, std::enable_if_t<std::is_pointer_v<T>>>>
  operator+(OffsetT Offset) const;

  template <typename OffsetT>
  friend std::enable_if_t<std::is_arithmetic_v<OffsetT>, VarTT<T, std::enable_if_t<std::is_pointer_v<T>>>>
  operator+(OffsetT Offset, const VarTT &Ptr) {
    return Ptr + Offset;
  }
};

// Non-member arithmetic operators for VarTT
template <typename T, typename U>
std::enable_if_t<std::is_arithmetic_v<T> && std::is_arithmetic_v<U>, 
                 VarTT<std::common_type_t<T, U>>>
operator+(const T &ConstValue, const VarTT<U> &Var);

template <typename T, typename U>
std::enable_if_t<std::is_arithmetic_v<T> && std::is_arithmetic_v<U>, 
                 VarTT<std::common_type_t<T, U>>>
operator-(const T &ConstValue, const VarTT<U> &Var);

template <typename T, typename U>
std::enable_if_t<std::is_arithmetic_v<T> && std::is_arithmetic_v<U>, 
                 VarTT<std::common_type_t<T, U>>>
operator*(const T &ConstValue, const VarTT<U> &Var);

template <typename T, typename U>
std::enable_if_t<std::is_arithmetic_v<T> && std::is_arithmetic_v<U>, 
                 VarTT<std::common_type_t<T, U>>>
operator/(const T &ConstValue, const VarTT<U> &Var);

template <typename T, typename U>
std::enable_if_t<std::is_arithmetic_v<T> && std::is_arithmetic_v<U>, 
                 VarTT<std::common_type_t<T, U>>>
operator%(const T &ConstValue, const VarTT<U> &Var);

// Math intrinsics for VarTT
template <typename T>
std::enable_if_t<std::is_same_v<T, float>, VarTT<T>>
powf(const VarTT<T> &L, const VarTT<T> &R);

template <typename T>
std::enable_if_t<std::is_same_v<T, float>, VarTT<T>>
sqrtf(const VarTT<T> &R);

template <typename T>
std::enable_if_t<std::is_arithmetic_v<T>, VarTT<T>>
min(const VarTT<T> &L, const VarTT<T> &R);
} // namespace proteus



#endif // PROTEUS_FRONTEND_VAR_HPP
