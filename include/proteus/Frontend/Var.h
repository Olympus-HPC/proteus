#ifndef PROTEUS_FRONTEND_VAR_H
#define PROTEUS_FRONTEND_VAR_H

#include "proteus/Frontend/TypeMap.h"
#include "proteus/Frontend/TypeTraits.h"
#include "proteus/Frontend/VarStorage.h"

#include <type_traits>

namespace proteus {

class FuncBase;

// Mixin that owns storage and exposes common helpers for Var specializations
template <typename StorageT> struct VarStorageOwner {
  FuncBase &Fn;
  std::unique_ptr<StorageT> Storage = nullptr;

  VarStorageOwner(std::unique_ptr<StorageT> StorageIn, FuncBase &FnIn)
      : Fn(FnIn), Storage(std::move(StorageIn)) {}

  VarStorageOwner(FuncBase &FnIn) : Fn(FnIn) {}

  // Storage accessor helpers
  llvm::Value *loadValue() const { return Storage->loadValue(); }

  void storeValue(llvm::Value *Val) { Storage->storeValue(Val); }

  llvm::Value *getSlot() const { return Storage->getSlot(); }
  llvm::Type *getSlotType() const { return Storage->getSlotType(); }
  llvm::Type *getValueType() const { return Storage->getValueType(); }
  llvm::Type *getAllocatedType() const { return Storage->getAllocatedType(); }
};

// Primary template declaration
template <typename T, typename = void> struct Var;

// Specialization for arithmetic types (including references to arithmetic
// types).
template <typename T>
struct Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>
    : public VarStorageOwner<VarStorage> {
  using ValueType = T;
  using ElemType = T;

  Var(std::unique_ptr<VarStorage> Storage, FuncBase &Fn)
      : VarStorageOwner<VarStorage>(std::move(Storage), Fn) {}

  // Conversion constructor from Var<U> where U can convert to T.
  template <typename U,
            typename = std::enable_if_t<std::is_convertible_v<U, T> &&
                                        (!std::is_same_v<U, T>)>>
  Var(const Var<U> &V);

  Var(const Var &V) : VarStorageOwner<VarStorage>(nullptr, V.Fn) {
    Storage = V.Storage->clone();
  }

  Var(Var &&V) : VarStorageOwner<VarStorage>(nullptr, V.Fn) {
    std::swap(Storage, V.Storage);
  }

  Var &operator=(Var &&V);

  // Assignment operators
  Var &operator=(const Var &V);

  template <typename U> Var &operator=(const Var<U> &V);

  template <typename U> Var &operator=(const U &ConstValue);

  Var<std::add_pointer_t<T>> getAddress();

  // Arithmetic operators
  template <typename U>
  Var<std::common_type_t<T, U>> operator+(const Var<U> &Other) const;

  template <typename U>
  std::enable_if_t<std::is_arithmetic_v<U>, Var<std::common_type_t<T, U>>>
  operator+(const U &ConstValue) const;

  template <typename U>
  Var<std::common_type_t<T, U>> operator-(const Var<U> &Other) const;

  template <typename U>
  std::enable_if_t<std::is_arithmetic_v<U>, Var<std::common_type_t<T, U>>>
  operator-(const U &ConstValue) const;

  template <typename U>
  Var<std::common_type_t<T, U>> operator*(const Var<U> &Other) const;

  template <typename U>
  std::enable_if_t<std::is_arithmetic_v<U>, Var<std::common_type_t<T, U>>>
  operator*(const U &ConstValue) const;

  template <typename U>
  Var<std::common_type_t<T, U>> operator/(const Var<U> &Other) const;

  template <typename U>
  std::enable_if_t<std::is_arithmetic_v<U>, Var<std::common_type_t<T, U>>>
  operator/(const U &ConstValue) const;

  template <typename U>
  Var<std::common_type_t<T, U>> operator%(const Var<U> &Other) const;

  template <typename U>
  std::enable_if_t<std::is_arithmetic_v<U>, Var<std::common_type_t<T, U>>>
  operator%(const U &ConstValue) const;

  // Unary operators
  Var<remove_cvref_t<T>> operator-() const;
  Var<bool> operator!() const;

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
  std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
  operator>(const Var<U> &Other) const;

  template <typename U>
  std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
  operator>=(const Var<U> &Other) const;

  template <typename U>
  std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
  operator<(const Var<U> &Other) const;

  template <typename U>
  std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
  operator<=(const Var<U> &Other) const;

  template <typename U>
  std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
  operator==(const Var<U> &Other) const;

  template <typename U>
  std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
  operator!=(const Var<U> &Other) const;

  template <typename U>
  std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
  operator>(const U &ConstValue) const;

  template <typename U>
  std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
  operator>=(const U &ConstValue) const;

  template <typename U>
  std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
  operator<(const U &ConstValue) const;

  template <typename U>
  std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
  operator<=(const U &ConstValue) const;

  template <typename U>
  std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
  operator==(const U &ConstValue) const;

  template <typename U>
  std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
  operator!=(const U &ConstValue) const;
};

// Specialization for array types
template <typename T>
struct Var<T, std::enable_if_t<std::is_array_v<T>>>
    : public VarStorageOwner<ArrayStorage> {
  using ValueType = T;
  using ElemType = std::remove_extent_t<T>;

  Var(std::unique_ptr<ArrayStorage> Storage, FuncBase &Fn)
      : VarStorageOwner<ArrayStorage>(std::move(Storage), Fn) {}

  Var<std::add_lvalue_reference_t<ElemType>> operator[](size_t Index);

  template <typename IdxT>
  std::enable_if_t<std::is_integral_v<IdxT>,
                   Var<std::add_lvalue_reference_t<ElemType>>>
  operator[](const Var<IdxT> &Index);

  Var<std::add_pointer_t<ValueType>> getAddress() const = delete;
};

// Specialization for pointer types (including references to pointers)
template <typename T>
struct Var<T, std::enable_if_t<is_pointer_unref_v<T>>>
    : public VarStorageOwner<PointerStorage> {
  using ValueType = T;
  using ElemType = std::remove_pointer_t<std::remove_reference_t<T>>;

  Var(std::unique_ptr<PointerStorage> Storage, FuncBase &Fn)
      : VarStorageOwner<PointerStorage>(std::move(Storage), Fn) {}

  // Load/store the pointer value itself from/to the pointer slot.
  llvm::Value *loadPointer() const { return this->Storage->loadPointer(); }
  void storePointer(llvm::Value *Ptr) { this->Storage->storePointer(Ptr); }

  Var<std::add_lvalue_reference_t<ElemType>> operator[](size_t Index);

  template <typename IdxT>
  std::enable_if_t<std::is_arithmetic_v<IdxT>,
                   Var<std::add_lvalue_reference_t<ElemType>>>
  operator[](const Var<IdxT> &Index);

  Var<std::add_lvalue_reference_t<ElemType>> operator*();

  Var<std::add_pointer_t<ValueType>> getAddress();

  template <typename OffsetT>
  std::enable_if_t<std::is_arithmetic_v<OffsetT>,
                   Var<T, std::enable_if_t<is_pointer_unref_v<T>>>>
  operator+(const Var<OffsetT> &Offset) const;

  template <typename OffsetT>
  std::enable_if_t<std::is_arithmetic_v<OffsetT>,
                   Var<T, std::enable_if_t<is_pointer_unref_v<T>>>>
  operator+(OffsetT Offset) const;

  template <typename OffsetT>
  friend std::enable_if_t<std::is_arithmetic_v<OffsetT>,
                          Var<T, std::enable_if_t<is_pointer_unref_v<T>>>>
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

} // namespace proteus

#endif // PROTEUS_FRONTEND_VAR_H
