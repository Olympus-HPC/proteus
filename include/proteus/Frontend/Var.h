#ifndef PROTEUS_FRONTEND_VAR_H
#define PROTEUS_FRONTEND_VAR_H

#include "proteus/Error.h"
#include "proteus/Frontend/LLVMCodeBuilder.h"
#include "proteus/Frontend/TypeMap.h"
#include "proteus/Frontend/TypeTraits.h"
#include "proteus/Frontend/VarStorage.h"

#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace proteus {

// Mixin that owns storage and exposes common helpers for Var specializations
template <typename StorageT> struct VarStorageOwner {
  LLVMCodeBuilder &CB;
  std::unique_ptr<StorageT> Storage = nullptr;

  VarStorageOwner(std::unique_ptr<StorageT> StorageIn, LLVMCodeBuilder &CBIn)
      : CB(CBIn), Storage(std::move(StorageIn)) {}

  VarStorageOwner(LLVMCodeBuilder &CBIn) : CB(CBIn) {}

  // Storage accessor helpers
  IRValue loadValue() const { return Storage->loadValue(); }

  void storeValue(IRValue Val) { Storage->storeValue(Val); }

  IRValue getSlot() const { return Storage->getSlot(); }
  IRType getValueType() const { return Storage->getValueType(); }
  IRType getAllocatedType() const { return Storage->getAllocatedType(); }
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

  Var(std::unique_ptr<VarStorage> Storage, LLVMCodeBuilder &CB)
      : VarStorageOwner<VarStorage>(std::move(Storage), CB) {}

  // Conversion constructor from Var<U> where U can convert to T.
  template <typename U,
            typename = std::enable_if_t<std::is_convertible_v<U, T> &&
                                        (!std::is_same_v<U, T>)>>
  Var(const Var<U> &V);

  Var(const Var &V) : VarStorageOwner<VarStorage>(nullptr, V.CB) {
    Storage = V.Storage->clone();
  }

  Var(Var &&V) : VarStorageOwner<VarStorage>(nullptr, V.CB) {
    std::swap(Storage, V.Storage);
  }

  Var &operator=(Var &&V);

  // Assignment operators
  Var &operator=(const Var &V);

  template <typename U> Var &operator=(const Var<U> &V);

  template <typename U> Var &operator=(const U &ConstValue);

  Var<std::add_pointer_t<T>> getAddress();

  /// Convert this variable's value to arithmetic type U and return a new Var
  /// holding the converted value. Preserves cv-qualifiers but drops
  /// references.
  template <typename U> auto convert() const;

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

  Var(std::unique_ptr<ArrayStorage> Storage, LLVMCodeBuilder &CB)
      : VarStorageOwner<ArrayStorage>(std::move(Storage), CB) {}

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

  Var(std::unique_ptr<PointerStorage> Storage, LLVMCodeBuilder &CB)
      : VarStorageOwner<PointerStorage>(std::move(Storage), CB) {}

  // Load/store the pointer value itself from/to the pointer slot.
  IRValue loadPointer() const { return this->Storage->loadPointer(); }
  void storePointer(IRValue Ptr) { this->Storage->storePointer(Ptr); }

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

// ---------------------------------------------------------------------------
// Free helper functions (type conversion)
// ---------------------------------------------------------------------------

// Value-level type conversion — internal implementation detail.
// Use Var::convert<U>() for user-facing type conversions.
namespace detail {
template <typename FromT, typename ToT>
IRValue convert(LLVMCodeBuilder &CB, IRValue V) {
  using From = remove_cvref_t<FromT>;
  using To = remove_cvref_t<ToT>;
  static_assert(std::is_arithmetic_v<From>, "From type must be arithmetic");
  static_assert(std::is_arithmetic_v<To>, "To type must be arithmetic");

  if constexpr (std::is_same_v<From, To>) {
    return V;
  }

  IRType DestTy = TypeMap<To>::get();

  if constexpr (std::is_integral_v<From> && std::is_floating_point_v<To>) {
    if constexpr (std::is_signed_v<From>) {
      return CB.createSIToFP(V, DestTy);
    }
    return CB.createUIToFP(V, DestTy);
  }

  if constexpr (std::is_floating_point_v<From> && std::is_integral_v<To>) {
    if constexpr (std::is_signed_v<To>) {
      return CB.createFPToSI(V, DestTy);
    }
    return CB.createFPToUI(V, DestTy);
  }

  if constexpr (std::is_integral_v<From> && std::is_integral_v<To>) {
    return CB.createIntCast(V, DestTy, std::is_signed_v<From>);
  }

  if constexpr (std::is_floating_point_v<From> &&
                std::is_floating_point_v<To>) {
    return CB.createFPCast(V, DestTy);
  }

  reportFatalError("Unsupported conversion");
}
} // namespace detail

// Allocate a new Var of type T using CB.
template <typename T>
Var<T> declVar(LLVMCodeBuilder &CB, const std::string &Name = "var") {
  static_assert(!std::is_array_v<T>, "Expected non-array type");
  static_assert(!std::is_reference_v<T>,
                "declVar does not support reference types");

  if constexpr (std::is_pointer_v<T>) {
    IRType ElemIRTy = *TypeMap<T>::getPointerElemType();
    return Var<T>{CB.createPointerStorage(Name, ElemIRTy), CB};
  } else {
    IRType AllocaIRTy = TypeMap<T>::get();
    return Var<T>{CB.createScalarStorage(Name, AllocaIRTy), CB};
  }
}

// Allocate and initialize a Var of type T.
template <typename T>
Var<T> defVar(LLVMCodeBuilder &CB, const T &Val,
              const std::string &Name = "var") {
  using RawT = std::remove_const_t<T>;
  Var<RawT> V = declVar<RawT>(CB, Name);
  V = Val;
  return Var<T>(V);
}

// ---------------------------------------------------------------------------
// Operator implementation helpers
// ---------------------------------------------------------------------------

template <typename T, typename U, typename IntOp, typename FPOp>
Var<std::common_type_t<remove_cvref_t<T>, remove_cvref_t<U>>>
binOp(const Var<T> &L, const Var<U> &R, IntOp IOp, FPOp FOp) {
  using CommonT = std::common_type_t<remove_cvref_t<T>, remove_cvref_t<U>>;

  LLVMCodeBuilder &CB = L.CB;
  if (&CB != &R.CB)
    reportFatalError("Variables should belong to the same function");

  IRValue LHS = detail::convert<T, CommonT>(CB, L.loadValue());
  IRValue RHS = detail::convert<U, CommonT>(CB, R.loadValue());

  IRValue Result;
  if constexpr (std::is_integral_v<CommonT>) {
    Result = IOp(CB, LHS, RHS);
  } else {
    Result = FOp(CB, LHS, RHS);
  }

  auto ResultVar = declVar<CommonT>(CB, "res.");
  ResultVar.storeValue(Result);

  return ResultVar;
}

template <typename T, typename U, typename IntOp, typename FPOp>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &
compoundAssignConst(Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &LHS,
                    const U &ConstValue, IntOp IOp, FPOp FOp) {
  static_assert(std::is_convertible_v<remove_cvref_t<U>, remove_cvref_t<T>>,
                "U must be convertible to T");

  IRType RHSType = TypeMap<remove_cvref_t<U>>::get();

  IRValue RHS;
  if constexpr (std::is_integral_v<remove_cvref_t<U>>) {
    RHS = LHS.CB.getConstantInt(RHSType, ConstValue);
  } else {
    RHS = LHS.CB.getConstantFP(RHSType, ConstValue);
  }

  IRValue LHSVal = LHS.loadValue();

  RHS = detail::convert<U, T>(LHS.CB, RHS);
  IRValue Result;

  if constexpr (std::is_integral_v<remove_cvref_t<T>>) {
    Result = IOp(LHS.CB, LHSVal, RHS);
  } else {
    static_assert(std::is_floating_point_v<remove_cvref_t<T>>,
                  "Unsupported type");
    Result = FOp(LHS.CB, LHSVal, RHS);
  }

  LHS.storeValue(Result);
  return LHS;
}

template <typename T, typename U, typename IntOp, typename FPOp>
Var<bool> cmpOp(const Var<T> &L, const Var<U> &R, IntOp IOp, FPOp FOp) {
  LLVMCodeBuilder &CB = L.CB;
  if (&CB != &R.CB)
    reportFatalError("Variables should belong to the same function");

  IRValue LHS = L.loadValue();
  IRValue RHS = detail::convert<U, T>(CB, R.loadValue());

  IRValue Result;
  if constexpr (std::is_integral_v<remove_cvref_t<T>>) {
    Result = IOp(CB, LHS, RHS);
  } else {
    static_assert(std::is_floating_point_v<remove_cvref_t<T>>,
                  "Unsupported type");
    Result = FOp(CB, LHS, RHS);
  }

  auto ResultVar = declVar<bool>(CB, "res.");
  ResultVar.storeValue(Result);

  return ResultVar;
}

// ---------------------------------------------------------------------------
// Var member operator implementations
// ---------------------------------------------------------------------------

template <typename T>
template <typename U>
auto Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::convert() const {
  using ResultT = std::remove_reference_t<U>;
  Var<ResultT> Res = declVar<ResultT>(this->CB, "convert.");
  IRValue Converted = detail::convert<T, U>(this->CB, this->loadValue());
  Res.storeValue(Converted);
  return Res;
}

template <typename T>
template <typename U, typename>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::Var(const Var<U> &V)
    : VarStorageOwner<VarStorage>(V.CB) {
  IRType TargetIRTy = TypeMap<remove_cvref_t<T>>::get();
  Storage = CB.createScalarStorage("conv.var", TargetIRTy);

  auto Converted = detail::convert<U, T>(CB, V.loadValue());
  storeValue(Converted);
}

template <typename T>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator=(const Var &V) {
  static_assert(is_mutable_v<T>, "Cannot assign to Var<const T>");
  storeValue(V.loadValue());
  return *this;
}

template <typename T>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator=(Var &&V) {
  static_assert(is_mutable_v<T>, "Cannot assign to Var<const T>");
  if (this->Storage == nullptr) {
    Storage = V.Storage->clone();
  } else {
    storeValue(V.loadValue());
  }
  return *this;
}

template <typename T>
Var<std::add_pointer_t<T>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::getAddress() {
  if constexpr (std::is_reference_v<T>) {
    auto *PtrStorage = static_cast<PointerStorage *>(Storage.get());
    IRValue PtrVal = PtrStorage->loadPointer();
    IRType ElemIRTy = PtrStorage->getValueType();
    unsigned AddrSpace = CB.getAddressSpaceFromValue(PtrVal);

    std::unique_ptr<PointerStorage> ResultStorage =
        CB.createPointerStorage("addr.ref.tmp", ElemIRTy, AddrSpace);
    CB.createStore(PtrVal, ResultStorage->getSlot());

    return Var<std::add_pointer_t<T>>(std::move(ResultStorage), CB);
  }

  IRValue Slot = getSlot();
  IRType ElemIRTy = getAllocatedType();
  unsigned AddrSpace = CB.getAddressSpaceFromValue(Slot);

  std::unique_ptr<PointerStorage> ResultStorage =
      CB.createPointerStorage("addr.tmp", ElemIRTy, AddrSpace);
  CB.createStore(Slot, ResultStorage->getSlot());

  return Var<std::add_pointer_t<T>>(std::move(ResultStorage), CB);
}

template <typename T>
template <typename U>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator=(
    const Var<U> &V) {
  static_assert(is_mutable_v<T>, "Cannot assign to Var<const T>");
  auto Converted = detail::convert<U, T>(CB, V.loadValue());
  storeValue(Converted);
  return *this;
}

template <typename T>
template <typename U>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator=(
    const U &ConstValue) {
  static_assert(is_mutable_v<T>, "Cannot assign to Var<const T>");
  static_assert(std::is_arithmetic_v<U>,
                "Can only assign arithmetic types to Var");

  IRType LHSType = getValueType();

  if (isIntegerKind(LHSType)) {
    storeValue(CB.getConstantInt(LHSType, ConstValue));
  } else if (isFloatingPointKind(LHSType)) {
    storeValue(CB.getConstantFP(LHSType, ConstValue));
  } else {
    reportFatalError("Unsupported type");
  }

  return *this;
}

template <typename T>
template <typename U>
Var<std::common_type_t<T, U>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator+(
    const Var<U> &Other) const {
  return binOp(
      *this, Other,
      [](LLVMCodeBuilder &CB, IRValue L, IRValue R) {
        return CB.createAdd(L, R);
      },
      [](LLVMCodeBuilder &CB, IRValue L, IRValue R) {
        return CB.createFAdd(L, R);
      });
}

template <typename T>
template <typename U>
Var<std::common_type_t<T, U>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator-(
    const Var<U> &Other) const {
  return binOp(
      *this, Other,
      [](LLVMCodeBuilder &CB, IRValue L, IRValue R) {
        return CB.createSub(L, R);
      },
      [](LLVMCodeBuilder &CB, IRValue L, IRValue R) {
        return CB.createFSub(L, R);
      });
}

template <typename T>
template <typename U>
Var<std::common_type_t<T, U>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator*(
    const Var<U> &Other) const {
  return binOp(
      *this, Other,
      [](LLVMCodeBuilder &CB, IRValue L, IRValue R) {
        return CB.createMul(L, R);
      },
      [](LLVMCodeBuilder &CB, IRValue L, IRValue R) {
        return CB.createFMul(L, R);
      });
}

template <typename T>
template <typename U>
Var<std::common_type_t<T, U>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator/(
    const Var<U> &Other) const {
  return binOp(
      *this, Other,
      [](LLVMCodeBuilder &CB, IRValue L, IRValue R) {
        return CB.createSDiv(L, R);
      },
      [](LLVMCodeBuilder &CB, IRValue L, IRValue R) {
        return CB.createFDiv(L, R);
      });
}

template <typename T>
template <typename U>
Var<std::common_type_t<T, U>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator%(
    const Var<U> &Other) const {
  return binOp(
      *this, Other,
      [](LLVMCodeBuilder &CB, IRValue L, IRValue R) {
        return CB.createSRem(L, R);
      },
      [](LLVMCodeBuilder &CB, IRValue L, IRValue R) {
        return CB.createFRem(L, R);
      });
}

// Arithmetic operators with ConstValue
template <typename T>
template <typename U>
std::enable_if_t<std::is_arithmetic_v<U>, Var<std::common_type_t<T, U>>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator+(
    const U &ConstValue) const {
  static_assert(std::is_arithmetic_v<U>,
                "Can only add arithmetic types to Var");
  Var<U> Tmp = defVar<U>(CB, ConstValue, "tmp.");
  return (*this) + Tmp;
}

template <typename T>
template <typename U>
std::enable_if_t<std::is_arithmetic_v<U>, Var<std::common_type_t<T, U>>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator-(
    const U &ConstValue) const {
  static_assert(std::is_arithmetic_v<U>,
                "Can only subtract arithmetic types from Var");
  Var<U> Tmp = defVar<U>(CB, ConstValue, "tmp.");
  return (*this) - Tmp;
}

template <typename T>
template <typename U>
std::enable_if_t<std::is_arithmetic_v<U>, Var<std::common_type_t<T, U>>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator*(
    const U &ConstValue) const {
  static_assert(std::is_arithmetic_v<U>,
                "Can only multiply Var by arithmetic types");
  Var<U> Tmp = defVar<U>(CB, ConstValue, "tmp.");
  return (*this) * Tmp;
}

template <typename T>
template <typename U>
std::enable_if_t<std::is_arithmetic_v<U>, Var<std::common_type_t<T, U>>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator/(
    const U &ConstValue) const {
  static_assert(std::is_arithmetic_v<U>,
                "Can only divide Var by arithmetic types");
  Var<U> Tmp = defVar<U>(CB, ConstValue, "tmp.");
  return (*this) / Tmp;
}

template <typename T>
template <typename U>
std::enable_if_t<std::is_arithmetic_v<U>, Var<std::common_type_t<T, U>>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator%(
    const U &ConstValue) const {
  static_assert(std::is_arithmetic_v<U>,
                "Can only modulo Var by arithmetic types");
  Var<U> Tmp = defVar<U>(CB, ConstValue, "tmp.");
  return (*this) % Tmp;
}

// Compound assignment operators for Var
template <typename T>
template <typename U>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator+=(
    const Var<U> &Other) {
  static_assert(is_mutable_v<T>, "Cannot use += on Var<const T>");
  auto Result = (*this) + Other;
  *this = Result;
  return *this;
}

template <typename T>
template <typename U>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator+=(
    const U &ConstValue) {
  static_assert(is_mutable_v<T>, "Cannot use += on Var<const T>");
  static_assert(std::is_arithmetic_v<U>,
                "Can only add arithmetic types to Var");
  return compoundAssignConst(
      *this, ConstValue,
      [](LLVMCodeBuilder &CB, IRValue L, IRValue R) {
        return CB.createAdd(L, R);
      },
      [](LLVMCodeBuilder &CB, IRValue L, IRValue R) {
        return CB.createFAdd(L, R);
      });
}

template <typename T>
template <typename U>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator-=(
    const Var<U> &Other) {
  static_assert(is_mutable_v<T>, "Cannot use -= on Var<const T>");
  auto Result = (*this) - Other;
  *this = Result;
  return *this;
}

template <typename T>
template <typename U>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator-=(
    const U &ConstValue) {
  static_assert(is_mutable_v<T>, "Cannot use -= on Var<const T>");
  static_assert(std::is_arithmetic_v<U>,
                "Can only subtract arithmetic types from Var");
  return compoundAssignConst(
      *this, ConstValue,
      [](LLVMCodeBuilder &CB, IRValue L, IRValue R) {
        return CB.createSub(L, R);
      },
      [](LLVMCodeBuilder &CB, IRValue L, IRValue R) {
        return CB.createFSub(L, R);
      });
}

template <typename T>
template <typename U>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator*=(
    const Var<U> &Other) {
  static_assert(is_mutable_v<T>, "Cannot use *= on Var<const T>");
  auto Result = (*this) * Other;
  *this = Result;
  return *this;
}

template <typename T>
template <typename U>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator*=(
    const U &ConstValue) {
  static_assert(is_mutable_v<T>, "Cannot use *= on Var<const T>");
  static_assert(std::is_arithmetic_v<U>,
                "Can only multiply Var by arithmetic types");
  return compoundAssignConst(
      *this, ConstValue,
      [](LLVMCodeBuilder &CB, IRValue L, IRValue R) {
        return CB.createMul(L, R);
      },
      [](LLVMCodeBuilder &CB, IRValue L, IRValue R) {
        return CB.createFMul(L, R);
      });
}

template <typename T>
template <typename U>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator/=(
    const Var<U> &Other) {
  static_assert(is_mutable_v<T>, "Cannot use /= on Var<const T>");
  auto Result = (*this) / Other;
  *this = Result;
  return *this;
}

template <typename T>
template <typename U>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator/=(
    const U &ConstValue) {
  static_assert(is_mutable_v<T>, "Cannot use /= on Var<const T>");
  static_assert(std::is_arithmetic_v<U>,
                "Can only divide Var by arithmetic types");
  return compoundAssignConst(
      *this, ConstValue,
      [](LLVMCodeBuilder &CB, IRValue L, IRValue R) {
        return CB.createSDiv(L, R);
      },
      [](LLVMCodeBuilder &CB, IRValue L, IRValue R) {
        return CB.createFDiv(L, R);
      });
}

template <typename T>
template <typename U>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator%=(
    const Var<U> &Other) {
  static_assert(is_mutable_v<T>, "Cannot use %= on Var<const T>");
  auto Result = (*this) % Other;
  *this = Result;
  return *this;
}

template <typename T>
template <typename U>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator%=(
    const U &ConstValue) {
  static_assert(is_mutable_v<T>, "Cannot use %= on Var<const T>");
  static_assert(std::is_arithmetic_v<U>,
                "Can only modulo Var by arithmetic types");
  return compoundAssignConst(
      *this, ConstValue,
      [](LLVMCodeBuilder &CB, IRValue L, IRValue R) {
        return CB.createSRem(L, R);
      },
      [](LLVMCodeBuilder &CB, IRValue L, IRValue R) {
        return CB.createFRem(L, R);
      });
}

template <typename T>
Var<remove_cvref_t<T>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator-() const {
  auto MinusOne = defVar<remove_cvref_t<T>>(
      CB, static_cast<remove_cvref_t<T>>(-1), "minus_one.");
  return MinusOne * (*this);
}

template <typename T>
Var<bool>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator!() const {
  IRValue V = loadValue();
  IRValue ResV;
  if constexpr (std::is_same_v<remove_cvref_t<T>, bool>) {
    ResV = CB.createNot(V);
  } else if constexpr (std::is_integral_v<remove_cvref_t<T>>) {
    IRValue Zero = CB.getConstantInt(getValueType(), 0);
    ResV = CB.createICmpEQ(V, Zero);
  } else {
    IRValue Zero = CB.getConstantFP(getValueType(), 0.0);
    ResV = CB.createFCmpOEQ(V, Zero);
  }
  auto Ret = declVar<bool>(CB, "not.");
  Ret.storeValue(ResV);
  return Ret;
}

template <typename T>
Var<std::add_lvalue_reference_t<std::remove_extent_t<T>>>
Var<T, std::enable_if_t<std::is_array_v<T>>>::operator[](size_t Index) {
  IRType ArrayIRTy = getAllocatedType();
  IRValue BasePointer = getSlot();

  IRValue GEP = CB.createConstInBoundsGEP2_64(ArrayIRTy, BasePointer, 0, Index);
  IRType ElemIRTy = getValueType();
  unsigned AddrSpace = CB.getAddressSpaceFromValue(BasePointer);

  std::unique_ptr<PointerStorage> ResultStorage =
      CB.createPointerStorage("elem.ptr", ElemIRTy, AddrSpace);
  CB.createStore(GEP, ResultStorage->getSlot());
  return Var<std::add_lvalue_reference_t<std::remove_extent_t<T>>>(
      std::move(ResultStorage), CB);
}

template <typename T>
template <typename IdxT>
std::enable_if_t<std::is_integral_v<IdxT>,
                 Var<std::add_lvalue_reference_t<std::remove_extent_t<T>>>>
Var<T, std::enable_if_t<std::is_array_v<T>>>::operator[](
    const Var<IdxT> &Index) {
  IRType ArrayIRTy = getAllocatedType();
  IRValue BasePointer = getSlot();

  IRValue IdxVal = Index.loadValue();
  IRValue Zero = CB.getConstantInt(Index.getValueType(), 0);
  IRValue GEP = CB.createInBoundsGEP(ArrayIRTy, BasePointer, {Zero, IdxVal});
  IRType ElemIRTy = getValueType();
  unsigned AddrSpace = CB.getAddressSpaceFromValue(BasePointer);

  std::unique_ptr<PointerStorage> ResultStorage =
      CB.createPointerStorage("elem.ptr", ElemIRTy, AddrSpace);
  CB.createStore(GEP, ResultStorage->getSlot());

  return Var<std::add_lvalue_reference_t<std::remove_extent_t<T>>>(
      std::move(ResultStorage), CB);
}

template <typename T>
Var<std::add_lvalue_reference_t<
    std::remove_pointer_t<std::remove_reference_t<T>>>>
Var<T, std::enable_if_t<is_pointer_unref_v<T>>>::operator[](size_t Index) {
  using ElemT = std::remove_pointer_t<std::remove_reference_t<T>>;
  IRType PointerElemIRTy = TypeMap<ElemT>::get();
  IRValue Ptr = loadPointer();
  IRValue GEP = CB.createConstInBoundsGEP1_64(PointerElemIRTy, Ptr, Index);
  unsigned AddrSpace = CB.getAddressSpaceFromValue(Ptr);

  std::unique_ptr<PointerStorage> ResultStorage =
      CB.createPointerStorage("elem.ptr", PointerElemIRTy, AddrSpace);
  CB.createStore(GEP, ResultStorage->getSlot());

  return Var<std::add_lvalue_reference_t<
      std::remove_pointer_t<std::remove_reference_t<T>>>>(
      std::move(ResultStorage), CB);
}

template <typename T>
template <typename IdxT>
std::enable_if_t<std::is_arithmetic_v<IdxT>,
                 Var<std::add_lvalue_reference_t<
                     std::remove_pointer_t<std::remove_reference_t<T>>>>>
Var<T, std::enable_if_t<is_pointer_unref_v<T>>>::operator[](
    const Var<IdxT> &Index) {
  using ElemT = std::remove_pointer_t<std::remove_reference_t<T>>;
  IRType PointeeIRTy = TypeMap<ElemT>::get();
  IRValue Ptr = loadPointer();
  IRValue IdxValue = Index.loadValue();
  IRValue GEP = CB.createInBoundsGEP(PointeeIRTy, Ptr, {IdxValue});
  unsigned AddrSpace = CB.getAddressSpaceFromValue(Ptr);

  std::unique_ptr<PointerStorage> ResultStorage =
      CB.createPointerStorage("elem.ptr", PointeeIRTy, AddrSpace);
  CB.createStore(GEP, ResultStorage->getSlot());

  return Var<std::add_lvalue_reference_t<
      std::remove_pointer_t<std::remove_reference_t<T>>>>(
      std::move(ResultStorage), CB);
}

template <typename T>
Var<std::add_lvalue_reference_t<
    std::remove_pointer_t<std::remove_reference_t<T>>>>
Var<T, std::enable_if_t<is_pointer_unref_v<T>>>::operator*() {
  return (*this)[0];
}

template <typename T>
Var<std::add_pointer_t<T>>
Var<T, std::enable_if_t<is_pointer_unref_v<T>>>::getAddress() {
  IRValue PtrVal = loadPointer();
  IRType ElemIRTy = getValueType();

  // Result holds a pointer-to-(pointer-to-ElemIRTy)
  IRType PointeePtrIRTy{IRTypeKind::Pointer, ElemIRTy.Signed, 0, ElemIRTy.Kind};

  std::unique_ptr<PointerStorage> ResultStorage =
      CB.createPointerStorage("addr.ptr.tmp", PointeePtrIRTy, 0);
  CB.createStore(PtrVal, ResultStorage->getSlot());

  return Var<std::add_pointer_t<T>>(std::move(ResultStorage), CB);
}

template <typename T>
template <typename OffsetT>
std::enable_if_t<std::is_arithmetic_v<OffsetT>,
                 Var<T, std::enable_if_t<is_pointer_unref_v<T>>>>
Var<T, std::enable_if_t<is_pointer_unref_v<T>>>::operator+(
    const Var<OffsetT> &Offset) const {
  IRValue OffsetVal = Offset.loadValue();
  IRValue IdxVal = detail::convert<OffsetT, int64_t>(CB, OffsetVal);

  IRValue BasePtr = loadPointer();
  IRType ElemIRTy = getValueType();

  IRValue GEP = CB.createInBoundsGEP(ElemIRTy, BasePtr, {IdxVal}, "ptr.add");

  unsigned AddrSpace = CB.getAddressSpaceFromValue(loadPointer());

  std::unique_ptr<PointerStorage> ResultStorage =
      CB.createPointerStorage("ptr.add.tmp", ElemIRTy, AddrSpace);
  CB.createStore(GEP, ResultStorage->getSlot());

  return Var<T, std::enable_if_t<is_pointer_unref_v<T>>>(
      std::move(ResultStorage), CB);
}

template <typename T>
template <typename OffsetT>
std::enable_if_t<std::is_arithmetic_v<OffsetT>,
                 Var<T, std::enable_if_t<is_pointer_unref_v<T>>>>
Var<T, std::enable_if_t<is_pointer_unref_v<T>>>::operator+(
    OffsetT Offset) const {
  IRValue IdxVal = CB.getConstantInt(IRType{IRTypeKind::Int64},
                                     static_cast<uint64_t>(Offset));

  IRValue BasePtr = loadPointer();
  IRType ElemIRTy = getValueType();

  IRValue GEP = CB.createInBoundsGEP(ElemIRTy, BasePtr, {IdxVal}, "ptr.add");

  unsigned AddrSpace = CB.getAddressSpaceFromValue(loadPointer());

  std::unique_ptr<PointerStorage> ResultStorage =
      CB.createPointerStorage("ptr.add.tmp", ElemIRTy, AddrSpace);
  CB.createStore(GEP, ResultStorage->getSlot());

  return Var<T, std::enable_if_t<is_pointer_unref_v<T>>>(
      std::move(ResultStorage), CB);
}

// Comparison operators for Var
template <typename T>
template <typename U>
std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator>(
    const Var<U> &Other) const {
  return cmpOp(
      *this, Other,
      [](LLVMCodeBuilder &CB, IRValue L, IRValue R) {
        return CB.createICmpSGT(L, R);
      },
      [](LLVMCodeBuilder &CB, IRValue L, IRValue R) {
        return CB.createFCmpOGT(L, R);
      });
}

template <typename T>
template <typename U>
std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator>=(
    const Var<U> &Other) const {
  return cmpOp(
      *this, Other,
      [](LLVMCodeBuilder &CB, IRValue L, IRValue R) {
        return CB.createICmpSGE(L, R);
      },
      [](LLVMCodeBuilder &CB, IRValue L, IRValue R) {
        return CB.createFCmpOGE(L, R);
      });
}

template <typename T>
template <typename U>
std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator<(
    const Var<U> &Other) const {
  return cmpOp(
      *this, Other,
      [](LLVMCodeBuilder &CB, IRValue L, IRValue R) {
        return CB.createICmpSLT(L, R);
      },
      [](LLVMCodeBuilder &CB, IRValue L, IRValue R) {
        return CB.createFCmpOLT(L, R);
      });
}

template <typename T>
template <typename U>
std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator<=(
    const Var<U> &Other) const {
  return cmpOp(
      *this, Other,
      [](LLVMCodeBuilder &CB, IRValue L, IRValue R) {
        return CB.createICmpSLE(L, R);
      },
      [](LLVMCodeBuilder &CB, IRValue L, IRValue R) {
        return CB.createFCmpOLE(L, R);
      });
}

template <typename T>
template <typename U>
std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator==(
    const Var<U> &Other) const {
  return cmpOp(
      *this, Other,
      [](LLVMCodeBuilder &CB, IRValue L, IRValue R) {
        return CB.createICmpEQ(L, R);
      },
      [](LLVMCodeBuilder &CB, IRValue L, IRValue R) {
        return CB.createFCmpOEQ(L, R);
      });
}

template <typename T>
template <typename U>
std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator!=(
    const Var<U> &Other) const {
  return cmpOp(
      *this, Other,
      [](LLVMCodeBuilder &CB, IRValue L, IRValue R) {
        return CB.createICmpNE(L, R);
      },
      [](LLVMCodeBuilder &CB, IRValue L, IRValue R) {
        return CB.createFCmpONE(L, R);
      });
}

template <typename T>
template <typename U>
std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator>(
    const U &ConstValue) const {
  Var<U> Tmp = defVar<U>(CB, ConstValue, "cmp.");
  return (*this) > Tmp;
}

template <typename T>
template <typename U>
std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator>=(
    const U &ConstValue) const {
  Var<U> Tmp = defVar<U>(CB, ConstValue, "cmp.");
  return (*this) >= Tmp;
}

template <typename T>
template <typename U>
std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator<(
    const U &ConstValue) const {
  Var<U> Tmp = defVar<U>(CB, ConstValue, "cmp.");
  return (*this) < Tmp;
}

template <typename T>
template <typename U>
std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator<=(
    const U &ConstValue) const {
  auto Tmp = defVar<U>(CB, ConstValue, "cmp.");
  return (*this) <= Tmp;
}

template <typename T>
template <typename U>
std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator==(
    const U &ConstValue) const {
  Var<U> Tmp = defVar<U>(CB, ConstValue, "cmp.");
  return (*this) == Tmp;
}

template <typename T>
template <typename U>
std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator!=(
    const U &ConstValue) const {
  auto Tmp = defVar<U>(CB, ConstValue, "cmp.");
  return (*this) != Tmp;
}

// Non-member arithmetic operators for Var
template <typename T, typename U>
std::enable_if_t<std::is_arithmetic_v<T> && std::is_arithmetic_v<U>,
                 Var<std::common_type_t<T, U>>>
operator+(const T &ConstValue, const Var<U> &V) {
  Var<T> Tmp = defVar<T>(V.CB, ConstValue, "tmp.");
  return Tmp + V;
}

template <typename T, typename U>
std::enable_if_t<std::is_arithmetic_v<T> && std::is_arithmetic_v<U>,
                 Var<std::common_type_t<T, U>>>
operator-(const T &ConstValue, const Var<U> &V) {
  using CommonType = std::common_type_t<T, U>;
  Var<CommonType> Tmp = defVar<CommonType>(V.CB, ConstValue, "tmp.");
  return Tmp - V;
}

template <typename T, typename U>
std::enable_if_t<std::is_arithmetic_v<T> && std::is_arithmetic_v<U>,
                 Var<std::common_type_t<T, U>>>
operator*(const T &ConstValue, const Var<U> &V) {
  Var<T> Tmp = defVar<T>(V.CB, ConstValue, "tmp.");
  return Tmp * V;
}

template <typename T, typename U>
std::enable_if_t<std::is_arithmetic_v<T> && std::is_arithmetic_v<U>,
                 Var<std::common_type_t<T, U>>>
operator/(const T &ConstValue, const Var<U> &V) {
  Var<T> Tmp = defVar<T>(V.CB, ConstValue, "tmp.");
  return Tmp / V;
}

template <typename T, typename U>
std::enable_if_t<std::is_arithmetic_v<T> && std::is_arithmetic_v<U>,
                 Var<std::common_type_t<T, U>>>
operator%(const T &ConstValue, const Var<U> &V) {
  Var<T> Tmp = defVar<T>(V.CB, ConstValue, "tmp.");
  return Tmp % V;
}

// ---------------------------------------------------------------------------
// Intrinsic emission helpers
// ---------------------------------------------------------------------------

// Helper struct to convert Var operands to a target type T.
template <typename T> struct IntrinsicOperandConverter {
  LLVMCodeBuilder &CB;

  template <typename U> IRValue operator()(const Var<U> &Operand) const {
    return detail::convert<U, T>(CB, Operand.loadValue());
  }
};

template <typename T, typename... Operands>
static Var<T> emitIntrinsic(const std::string &IntrinsicName,
                            const Operands &...Ops) {
  static_assert(sizeof...(Ops) > 0, "Intrinsic requires at least one operand");

  LLVMCodeBuilder &CB = std::get<0>(std::tie(Ops...)).CB;
  auto CheckFn = [&CB](const auto &Operand) {
    if (&Operand.CB != &CB)
      reportFatalError("Variables should belong to the same function");
  };
  (CheckFn(Ops), ...);

  IntrinsicOperandConverter<T> ConvertOperand{CB};

  IRType ResultIRTy = TypeMap<T>::get();
  std::vector<IRType> ArgTys(sizeof...(Ops), ResultIRTy);
  IRValue Call = CB.createCall(IntrinsicName, ResultIRTy, ArgTys,
                               {ConvertOperand(Ops)...});

  auto ResultVar = declVar<T>(CB, "res.");
  ResultVar.storeValue(Call);
  return ResultVar;
}

// ---------------------------------------------------------------------------
// Math intrinsics for Var
// ---------------------------------------------------------------------------

template <typename T> Var<float> powf(const Var<float> &L, const Var<T> &R) {
  static_assert(std::is_convertible_v<T, float>,
                "powf requires floating-point type");

  auto RFloat = R.template convert<float>();
  std::string IntrinsicName = "llvm.pow.f32";
#if PROTEUS_ENABLE_CUDA
  if (L.CB.getTargetModel() == TargetModelType::CUDA)
    IntrinsicName = "__nv_powf";
#endif

  return emitIntrinsic<float>(IntrinsicName, L, RFloat);
}

template <typename T> Var<float> sqrtf(const Var<T> &R) {
  static_assert(std::is_convertible_v<T, float>,
                "sqrtf requires floating-point type");

  auto RFloat = R.template convert<float>();
  std::string IntrinsicName = "llvm.sqrt.f32";
#if PROTEUS_ENABLE_CUDA
  if (R.CB.getTargetModel() == TargetModelType::CUDA)
    IntrinsicName = "__nv_sqrtf";
#endif

  return emitIntrinsic<float>(IntrinsicName, RFloat);
}

template <typename T> Var<float> expf(const Var<T> &R) {
  static_assert(std::is_convertible_v<T, float>,
                "expf requires floating-point type");

  auto RFloat = R.template convert<float>();
  std::string IntrinsicName = "llvm.exp.f32";
#if PROTEUS_ENABLE_CUDA
  if (R.CB.getTargetModel() == TargetModelType::CUDA)
    IntrinsicName = "__nv_expf";
#endif

  return emitIntrinsic<float>(IntrinsicName, RFloat);
}

template <typename T> Var<float> sinf(const Var<T> &R) {
  static_assert(std::is_convertible_v<T, float>,
                "sinf requires floating-point type");

  auto RFloat = R.template convert<float>();
  std::string IntrinsicName = "llvm.sin.f32";
#if PROTEUS_ENABLE_CUDA
  if (R.CB.getTargetModel() == TargetModelType::CUDA)
    IntrinsicName = "__nv_sinf";
#endif

  return emitIntrinsic<float>(IntrinsicName, RFloat);
}

template <typename T> Var<float> cosf(const Var<T> &R) {
  static_assert(std::is_convertible_v<T, float>,
                "cosf requires floating-point type");

  auto RFloat = R.template convert<float>();
  std::string IntrinsicName = "llvm.cos.f32";
#if PROTEUS_ENABLE_CUDA
  if (R.CB.getTargetModel() == TargetModelType::CUDA)
    IntrinsicName = "__nv_cosf";
#endif

  return emitIntrinsic<float>(IntrinsicName, RFloat);
}

template <typename T> Var<float> fabs(const Var<T> &R) {
  static_assert(std::is_convertible_v<T, float>,
                "fabs requires floating-point type");

  auto RFloat = R.template convert<float>();
  std::string IntrinsicName = "llvm.fabs.f32";
#if PROTEUS_ENABLE_CUDA
  if (R.CB.getTargetModel() == TargetModelType::CUDA)
    IntrinsicName = "__nv_fabsf";
#endif

  return emitIntrinsic<float>(IntrinsicName, RFloat);
}

template <typename T> Var<float> truncf(const Var<T> &R) {
  static_assert(std::is_convertible_v<T, float>,
                "truncf requires floating-point type");

  auto RFloat = R.template convert<float>();
  std::string IntrinsicName = "llvm.trunc.f32";
#if PROTEUS_ENABLE_CUDA
  if (R.CB.getTargetModel() == TargetModelType::CUDA)
    IntrinsicName = "__nv_truncf";
#endif

  return emitIntrinsic<float>(IntrinsicName, RFloat);
}

template <typename T> Var<float> logf(const Var<T> &R) {
  static_assert(std::is_convertible_v<T, float>,
                "logf requires floating-point type");

  auto RFloat = R.template convert<float>();
  std::string IntrinsicName = "llvm.log.f32";
#if PROTEUS_ENABLE_CUDA
  if (R.CB.getTargetModel() == TargetModelType::CUDA)
    IntrinsicName = "__nv_logf";
#endif

  return emitIntrinsic<float>(IntrinsicName, RFloat);
}

template <typename T> Var<float> absf(const Var<T> &R) {
  static_assert(std::is_convertible_v<T, float>,
                "absf requires floating-point type");

  auto RFloat = R.template convert<float>();
  std::string IntrinsicName = "llvm.fabs.f32";
#if PROTEUS_ENABLE_CUDA
  if (R.CB.getTargetModel() == TargetModelType::CUDA)
    IntrinsicName = "__nv_fabsf";
#endif

  return emitIntrinsic<float>(IntrinsicName, RFloat);
}

template <typename T>
std::enable_if_t<is_arithmetic_unref_v<T>, Var<remove_cvref_t<T>>>
min(const Var<T> &L, const Var<T> &R) {
  LLVMCodeBuilder &CB = L.CB;
  if (&CB != &R.CB)
    reportFatalError("Variables should belong to the same function");

  auto ResultVar = declVar<remove_cvref_t<T>>(CB, "min_res");
  ResultVar = R;
  auto CondVar = L < R;
  CB.beginIf(CondVar.loadValue(), __builtin_FILE(), __builtin_LINE());
  { ResultVar = L; }
  CB.endIf();
  return ResultVar;
}

template <typename T>
std::enable_if_t<is_arithmetic_unref_v<T>, Var<remove_cvref_t<T>>>
max(const Var<T> &L, const Var<T> &R) {
  LLVMCodeBuilder &CB = L.CB;
  if (&CB != &R.CB)
    reportFatalError("Variables should belong to the same function");

  auto ResultVar = declVar<remove_cvref_t<T>>(CB, "max_res");
  ResultVar = R;
  auto CondVar = L > R;
  CB.beginIf(CondVar.loadValue(), __builtin_FILE(), __builtin_LINE());
  { ResultVar = L; }
  CB.endIf();
  return ResultVar;
}

} // namespace proteus

#endif // PROTEUS_FRONTEND_VAR_H
