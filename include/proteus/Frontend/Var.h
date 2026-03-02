#ifndef PROTEUS_FRONTEND_VAR_H
#define PROTEUS_FRONTEND_VAR_H

#include "proteus/Error.h"
#include "proteus/Frontend/CodeBuilder.h"
#include "proteus/Frontend/TypeMap.h"
#include "proteus/Frontend/TypeTraits.h"

#include <string>
#include <type_traits>

namespace proteus {

// Primary template declaration
template <typename T, typename = void> struct Var;

// Specialization for arithmetic types (including references to arithmetic
// types).
template <typename T>
struct Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> {
  CodeBuilder &CB;
  IRValue *Slot;
  IRType ValueTy;
  IRType AllocTy;
  unsigned AddrSpace = 0;

  using ValueType = T;
  using ElemType = T;

  Var(VarAlloc A, CodeBuilder &CBIn)
      : CB(CBIn), Slot(A.Slot), ValueTy(A.ValueTy), AllocTy(A.AllocTy),
        AddrSpace(A.AddrSpace) {}

  // Conversion constructor from Var<U> where U can convert to T.
  template <typename U,
            typename = std::enable_if_t<std::is_convertible_v<U, T> &&
                                        (!std::is_same_v<U, T>)>>
  Var(const Var<U> &V);

  // Copy constructor: aliases the same alloca slot. This is effectively a
  // "shallow copy" that creates another Var handle to the same storage.
  Var(const Var &V) = default;
  Var(Var &&) = default;

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

  // Load / store helpers.
  IRValue *loadValue() const {
    if constexpr (std::is_reference_v<T>)
      return CB.loadFromPointee(Slot, AllocTy, ValueTy);
    else
      return CB.loadScalar(Slot, ValueTy);
  }
  void storeValue(IRValue *Val) {
    if constexpr (std::is_reference_v<T>)
      CB.storeToPointee(Slot, AllocTy, Val);
    else
      CB.storeScalar(Slot, Val);
  }
  IRValue *getSlot() const { return Slot; }
  IRType getValueType() const { return ValueTy; }
  IRType getAllocatedType() const { return AllocTy; }

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
template <typename T> struct Var<T, std::enable_if_t<std::is_array_v<T>>> {
  CodeBuilder &CB;
  IRValue *Slot;
  IRType ValueTy; ///< Element type.
  IRType AllocTy; ///< Array type.
  unsigned AddrSpace = 0;

  using ValueType = T;
  using ElemType = std::remove_extent_t<T>;

  Var(VarAlloc A, CodeBuilder &CBIn)
      : CB(CBIn), Slot(A.Slot), ValueTy(A.ValueTy), AllocTy(A.AllocTy),
        AddrSpace(A.AddrSpace) {}

  IRValue *getSlot() const { return Slot; }
  IRType getValueType() const { return ValueTy; }
  IRType getAllocatedType() const { return AllocTy; }

  // Load / store: loading an entire array is not supported.
  IRValue *loadValue() const {
    reportFatalError("Cannot load entire array as a value");
    return nullptr;
  }
  void storeValue(IRValue *) {
    reportFatalError("Cannot store value to entire array");
  }

  Var<std::add_lvalue_reference_t<ElemType>> operator[](size_t Index);

  template <typename IdxT>
  std::enable_if_t<std::is_integral_v<IdxT>,
                   Var<std::add_lvalue_reference_t<ElemType>>>
  operator[](const Var<IdxT> &Index);

  Var<std::add_pointer_t<ValueType>> getAddress() const = delete;
};

// Specialization for pointer types (including references to pointers)
template <typename T> struct Var<T, std::enable_if_t<is_pointer_unref_v<T>>> {
  CodeBuilder &CB;
  IRValue *Slot;
  IRType ValueTy; ///< Pointee (element) type.
  IRType AllocTy; ///< Type of the pointer alloca.
  unsigned AddrSpace = 0;

  using ValueType = T;
  using ElemType = std::remove_pointer_t<std::remove_reference_t<T>>;

  Var(VarAlloc A, CodeBuilder &CBIn)
      : CB(CBIn), Slot(A.Slot), ValueTy(A.ValueTy), AllocTy(A.AllocTy),
        AddrSpace(A.AddrSpace) {}

  IRValue *getSlot() const { return Slot; }
  IRType getValueType() const { return ValueTy; }
  IRType getAllocatedType() const { return AllocTy; }

  // Load / store the pointer value itself from/to the pointer slot.
  IRValue *loadAddress() const { return CB.loadAddress(Slot, AllocTy); }
  void storeAddress(IRValue *Ptr) { CB.storeAddress(Slot, Ptr); }

  // Load / store through the pointer (dereference).
  IRValue *loadValue() const {
    return CB.loadFromPointee(Slot, AllocTy, ValueTy);
  }
  void storeValue(IRValue *Val) { CB.storeToPointee(Slot, AllocTy, Val); }

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
IRValue *convert(CodeBuilder &CB, IRValue *V) {
  using From = remove_cvref_t<FromT>;
  using To = remove_cvref_t<ToT>;
  static_assert(std::is_arithmetic_v<From>, "From type must be arithmetic");
  static_assert(std::is_arithmetic_v<To>, "To type must be arithmetic");

  if constexpr (std::is_same_v<From, To>)
    return V;
  return CB.createCast(V, TypeMap<From>::get(), TypeMap<To>::get());
}
} // namespace detail

// Allocate a new Var of type T using CB.
template <typename T>
Var<T> declVar(CodeBuilder &CB, const std::string &Name = "var") {
  static_assert(!std::is_array_v<T>, "Expected non-array type");
  static_assert(!std::is_reference_v<T>,
                "declVar does not support reference types");

  if constexpr (std::is_pointer_v<T>) {
    IRType ElemIRTy = *TypeMap<T>::getPointerElemType();
    return Var<T>{CB.allocPointer(Name, ElemIRTy), CB};
  } else {
    IRType AllocaIRTy = TypeMap<T>::get();
    return Var<T>{CB.allocScalar(Name, AllocaIRTy), CB};
  }
}

// Allocate and initialize a Var of type T.
template <typename T>
Var<T> defVar(CodeBuilder &CB, const T &Val, const std::string &Name = "var") {
  using RawT = std::remove_const_t<T>;
  Var<RawT> V = declVar<RawT>(CB, Name);
  V = Val;
  return Var<T>(V);
}

// ---------------------------------------------------------------------------
// Operator implementation helpers
// ---------------------------------------------------------------------------

template <typename T, typename U>
Var<std::common_type_t<remove_cvref_t<T>, remove_cvref_t<U>>>
binOp(const Var<T> &L, const Var<U> &R, ArithOp Op) {
  using CommonT = std::common_type_t<remove_cvref_t<T>, remove_cvref_t<U>>;

  CodeBuilder &CB = L.CB;
  if (&CB != &R.CB)
    reportFatalError("Variables should belong to the same function");

  IRValue *LHS = detail::convert<T, CommonT>(CB, L.loadValue());
  IRValue *RHS = detail::convert<U, CommonT>(CB, R.loadValue());

  IRValue *Result = CB.createArith(Op, LHS, RHS, TypeMap<CommonT>::get());

  auto ResultVar = declVar<CommonT>(CB, "res.");
  ResultVar.storeValue(Result);

  return ResultVar;
}

template <typename T, typename U>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &
compoundAssignConst(Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &LHS,
                    const U &ConstValue, ArithOp Op) {
  static_assert(std::is_convertible_v<remove_cvref_t<U>, remove_cvref_t<T>>,
                "U must be convertible to T");

  IRType RHSType = TypeMap<remove_cvref_t<U>>::get();

  IRValue *RHS = nullptr;
  if constexpr (std::is_integral_v<remove_cvref_t<U>>) {
    RHS = LHS.CB.getConstantInt(RHSType, ConstValue);
  } else {
    RHS = LHS.CB.getConstantFP(RHSType, ConstValue);
  }

  IRValue *LHSVal = LHS.loadValue();

  RHS = detail::convert<U, T>(LHS.CB, RHS);
  IRValue *Result =
      LHS.CB.createArith(Op, LHSVal, RHS, TypeMap<remove_cvref_t<T>>::get());

  LHS.storeValue(Result);
  return LHS;
}

template <typename T, typename U>
Var<bool> cmpOp(const Var<T> &L, const Var<U> &R, CmpOp Op) {
  CodeBuilder &CB = L.CB;
  if (&CB != &R.CB)
    reportFatalError("Variables should belong to the same function");

  IRValue *LHS = L.loadValue();
  IRValue *RHS = detail::convert<U, T>(CB, R.loadValue());

  IRValue *Result =
      CB.createCmp(Op, LHS, RHS, TypeMap<remove_cvref_t<T>>::get());

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
  IRValue *Converted = detail::convert<T, U>(this->CB, this->loadValue());
  Res.storeValue(Converted);
  return Res;
}

template <typename T>
template <typename U, typename>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::Var(const Var<U> &V)
    : Var(V.CB.allocScalar("conv.var", TypeMap<remove_cvref_t<T>>::get()),
          V.CB) {
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
  storeValue(V.loadValue());
  return *this;
}

template <typename T>
Var<std::add_pointer_t<T>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::getAddress() {
  if constexpr (std::is_reference_v<T>) {
    // For a reference Var the slot holds a pointer; load that pointer and
    // expose it as the address.
    IRValue *PtrVal = CB.loadAddress(Slot, AllocTy);
    auto A = CB.allocPointer("addr.ref.tmp", ValueTy, AddrSpace);
    CB.storeAddress(A.Slot, PtrVal);
    return Var<std::add_pointer_t<T>>(A, CB);
  }

  auto A = CB.allocPointer("addr.tmp", AllocTy, AddrSpace);
  CB.storeAddress(A.Slot, Slot);
  return Var<std::add_pointer_t<T>>(A, CB);
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
  return binOp(*this, Other, ArithOp::Add);
}

template <typename T>
template <typename U>
Var<std::common_type_t<T, U>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator-(
    const Var<U> &Other) const {
  return binOp(*this, Other, ArithOp::Sub);
}

template <typename T>
template <typename U>
Var<std::common_type_t<T, U>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator*(
    const Var<U> &Other) const {
  return binOp(*this, Other, ArithOp::Mul);
}

template <typename T>
template <typename U>
Var<std::common_type_t<T, U>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator/(
    const Var<U> &Other) const {
  return binOp(*this, Other, ArithOp::Div);
}

template <typename T>
template <typename U>
Var<std::common_type_t<T, U>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator%(
    const Var<U> &Other) const {
  return binOp(*this, Other, ArithOp::Rem);
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
  return compoundAssignConst(*this, ConstValue, ArithOp::Add);
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
  return compoundAssignConst(*this, ConstValue, ArithOp::Sub);
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
  return compoundAssignConst(*this, ConstValue, ArithOp::Mul);
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
  return compoundAssignConst(*this, ConstValue, ArithOp::Div);
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
  return compoundAssignConst(*this, ConstValue, ArithOp::Rem);
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
  IRValue *V = loadValue();
  IRValue *ResV = nullptr;
  if constexpr (std::is_same_v<remove_cvref_t<T>, bool>) {
    ResV = CB.createNot(V);
  } else if constexpr (std::is_integral_v<remove_cvref_t<T>>) {
    IRValue *Zero = CB.getConstantInt(getValueType(), 0);
    ResV = CB.createCmp(CmpOp::EQ, V, Zero, getValueType());
  } else {
    IRValue *Zero = CB.getConstantFP(getValueType(), 0.0);
    ResV = CB.createCmp(CmpOp::EQ, V, Zero, getValueType());
  }
  auto Ret = declVar<bool>(CB, "not.");
  Ret.storeValue(ResV);
  return Ret;
}

template <typename T>
Var<std::add_lvalue_reference_t<std::remove_extent_t<T>>>
Var<T, std::enable_if_t<std::is_array_v<T>>>::operator[](size_t Index) {
  auto A = CB.getElementPtr(Slot, AllocTy, Index, ValueTy);
  return Var<std::add_lvalue_reference_t<std::remove_extent_t<T>>>(A, CB);
}

template <typename T>
template <typename IdxT>
std::enable_if_t<std::is_integral_v<IdxT>,
                 Var<std::add_lvalue_reference_t<std::remove_extent_t<T>>>>
Var<T, std::enable_if_t<std::is_array_v<T>>>::operator[](
    const Var<IdxT> &Index) {
  auto A = CB.getElementPtr(Slot, AllocTy, Index.loadValue(), ValueTy);
  return Var<std::add_lvalue_reference_t<std::remove_extent_t<T>>>(A, CB);
}

template <typename T>
Var<std::add_lvalue_reference_t<
    std::remove_pointer_t<std::remove_reference_t<T>>>>
Var<T, std::enable_if_t<is_pointer_unref_v<T>>>::operator[](size_t Index) {
  using ElemT = std::remove_pointer_t<std::remove_reference_t<T>>;
  IRType ElemIRTy = TypeMap<ElemT>::get();
  IRValue *Ptr = CB.loadAddress(Slot, AllocTy);
  auto A = CB.getElementPtr(Ptr, AllocTy, Index, ElemIRTy);
  return Var<std::add_lvalue_reference_t<
      std::remove_pointer_t<std::remove_reference_t<T>>>>(A, CB);
}

template <typename T>
template <typename IdxT>
std::enable_if_t<std::is_arithmetic_v<IdxT>,
                 Var<std::add_lvalue_reference_t<
                     std::remove_pointer_t<std::remove_reference_t<T>>>>>
Var<T, std::enable_if_t<is_pointer_unref_v<T>>>::operator[](
    const Var<IdxT> &Index) {
  using ElemT = std::remove_pointer_t<std::remove_reference_t<T>>;
  IRType ElemIRTy = TypeMap<ElemT>::get();
  IRValue *Ptr = CB.loadAddress(Slot, AllocTy);
  auto A = CB.getElementPtr(Ptr, AllocTy, Index.loadValue(), ElemIRTy);
  return Var<std::add_lvalue_reference_t<
      std::remove_pointer_t<std::remove_reference_t<T>>>>(A, CB);
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
  IRValue *PtrVal = CB.loadAddress(Slot, AllocTy);
  IRType PointeePtrIRTy{IRTypeKind::Pointer, ValueTy.Signed, 0, ValueTy.Kind};

  auto A = CB.allocPointer("addr.ptr.tmp", PointeePtrIRTy, 0);
  CB.storeAddress(A.Slot, PtrVal);
  return Var<std::add_pointer_t<T>>(A, CB);
}

template <typename T>
template <typename OffsetT>
std::enable_if_t<std::is_arithmetic_v<OffsetT>,
                 Var<T, std::enable_if_t<is_pointer_unref_v<T>>>>
Var<T, std::enable_if_t<is_pointer_unref_v<T>>>::operator+(
    const Var<OffsetT> &Offset) const {
  IRValue *IdxVal = detail::convert<OffsetT, int64_t>(CB, Offset.loadValue());
  IRValue *BasePtr = CB.loadAddress(Slot, AllocTy);
  auto A = CB.getElementPtr(BasePtr, AllocTy, IdxVal, ValueTy);
  return Var<T, std::enable_if_t<is_pointer_unref_v<T>>>(A, CB);
}

template <typename T>
template <typename OffsetT>
std::enable_if_t<std::is_arithmetic_v<OffsetT>,
                 Var<T, std::enable_if_t<is_pointer_unref_v<T>>>>
Var<T, std::enable_if_t<is_pointer_unref_v<T>>>::operator+(
    OffsetT Offset) const {
  IRValue *IdxVal = CB.getConstantInt(IRType{IRTypeKind::Int64},
                                      static_cast<uint64_t>(Offset));
  IRValue *BasePtr = CB.loadAddress(Slot, AllocTy);
  auto A = CB.getElementPtr(BasePtr, AllocTy, IdxVal, ValueTy);
  return Var<T, std::enable_if_t<is_pointer_unref_v<T>>>(A, CB);
}

// Comparison operators for Var
template <typename T>
template <typename U>
std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator>(
    const Var<U> &Other) const {
  return cmpOp(*this, Other, CmpOp::GT);
}

template <typename T>
template <typename U>
std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator>=(
    const Var<U> &Other) const {
  return cmpOp(*this, Other, CmpOp::GE);
}

template <typename T>
template <typename U>
std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator<(
    const Var<U> &Other) const {
  return cmpOp(*this, Other, CmpOp::LT);
}

template <typename T>
template <typename U>
std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator<=(
    const Var<U> &Other) const {
  return cmpOp(*this, Other, CmpOp::LE);
}

template <typename T>
template <typename U>
std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator==(
    const Var<U> &Other) const {
  return cmpOp(*this, Other, CmpOp::EQ);
}

template <typename T>
template <typename U>
std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator!=(
    const Var<U> &Other) const {
  return cmpOp(*this, Other, CmpOp::NE);
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
  CodeBuilder &CB;

  template <typename U> IRValue *operator()(const Var<U> &Operand) const {
    return detail::convert<U, T>(CB, Operand.loadValue());
  }
};

template <typename T, typename... Operands>
static Var<T> emitIntrinsic(const std::string &IntrinsicName,
                            const Operands &...Ops) {
  static_assert(sizeof...(Ops) > 0, "Intrinsic requires at least one operand");

  CodeBuilder &CB = std::get<0>(std::tie(Ops...)).CB;
  auto CheckFn = [&CB](const auto &Operand) {
    if (&Operand.CB != &CB)
      reportFatalError("Variables should belong to the same function");
  };
  (CheckFn(Ops), ...);

  IntrinsicOperandConverter<T> ConvertOperand{CB};

  IRType ResultIRTy = TypeMap<T>::get();
  IRValue *Call =
      CB.emitIntrinsic(IntrinsicName, ResultIRTy, {ConvertOperand(Ops)...});

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

  return emitIntrinsic<float>("powf", L, RFloat);
}

template <typename T> Var<float> sqrtf(const Var<T> &R) {
  static_assert(std::is_convertible_v<T, float>,
                "sqrtf requires floating-point type");

  auto RFloat = R.template convert<float>();
  return emitIntrinsic<float>("sqrtf", RFloat);
}

template <typename T> Var<float> expf(const Var<T> &R) {
  static_assert(std::is_convertible_v<T, float>,
                "expf requires floating-point type");

  auto RFloat = R.template convert<float>();
  return emitIntrinsic<float>("expf", RFloat);
}

template <typename T> Var<float> sinf(const Var<T> &R) {
  static_assert(std::is_convertible_v<T, float>,
                "sinf requires floating-point type");

  auto RFloat = R.template convert<float>();
  return emitIntrinsic<float>("sinf", RFloat);
}

template <typename T> Var<float> cosf(const Var<T> &R) {
  static_assert(std::is_convertible_v<T, float>,
                "cosf requires floating-point type");

  auto RFloat = R.template convert<float>();
  return emitIntrinsic<float>("cosf", RFloat);
}

template <typename T> Var<float> fabs(const Var<T> &R) {
  static_assert(std::is_convertible_v<T, float>,
                "fabs requires floating-point type");

  auto RFloat = R.template convert<float>();
  return emitIntrinsic<float>("fabsf", RFloat);
}

template <typename T> Var<float> truncf(const Var<T> &R) {
  static_assert(std::is_convertible_v<T, float>,
                "truncf requires floating-point type");

  auto RFloat = R.template convert<float>();
  return emitIntrinsic<float>("truncf", RFloat);
}

template <typename T> Var<float> logf(const Var<T> &R) {
  static_assert(std::is_convertible_v<T, float>,
                "logf requires floating-point type");

  auto RFloat = R.template convert<float>();
  return emitIntrinsic<float>("logf", RFloat);
}

template <typename T> Var<float> absf(const Var<T> &R) {
  static_assert(std::is_convertible_v<T, float>,
                "absf requires floating-point type");

  auto RFloat = R.template convert<float>();
  return emitIntrinsic<float>("absf", RFloat);
}

template <typename T>
std::enable_if_t<is_arithmetic_unref_v<T>, Var<remove_cvref_t<T>>>
min(const Var<T> &L, const Var<T> &R) {
  CodeBuilder &CB = L.CB;
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
  CodeBuilder &CB = L.CB;
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
