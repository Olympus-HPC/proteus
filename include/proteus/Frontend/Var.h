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

// ---------------------------------------------------------------------------
// Free helper functions (type conversion)
// ---------------------------------------------------------------------------

// Value-level type conversion (mirrors FuncBase::convert<FromT,ToT>).
template <typename FromT, typename ToT>
llvm::Value *convert(LLVMCodeBuilder &CB, llvm::Value *V) {
  using From = remove_cvref_t<FromT>;
  using To = remove_cvref_t<ToT>;
  static_assert(std::is_arithmetic_v<From>, "From type must be arithmetic");
  static_assert(std::is_arithmetic_v<To>, "To type must be arithmetic");

  auto &Ctx = CB.getContext();

  if constexpr (std::is_same_v<From, To>) {
    return V;
  }

  llvm::Type *DestTy = TypeMap<To>::get(Ctx);

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

// Allocate a new Var of type T using CB.
template <typename T>
Var<T> declVar(LLVMCodeBuilder &CB, const std::string &Name = "var") {
  static_assert(!std::is_array_v<T>, "Expected non-array type");
  static_assert(!std::is_reference_v<T>,
                "declVar does not support reference types");

  auto &Ctx = CB.getContext();
  llvm::Type *AllocaTy = TypeMap<T>::get(Ctx);

  if constexpr (std::is_pointer_v<T>) {
    llvm::Type *PtrElemTy = TypeMap<T>::getPointerElemType(Ctx);
    return Var<T>{CB.createPointerStorage(Name, AllocaTy, PtrElemTy), CB};
  } else {
    return Var<T>{CB.createScalarStorage(Name, AllocaTy), CB};
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

// Var-level conversion: allocate a new Var<U> and convert V into it.
template <typename U, typename T>
std::enable_if_t<std::is_convertible_v<std::remove_reference_t<T>,
                                       std::remove_reference_t<U>>,
                 Var<std::remove_reference_t<U>>>
convertVar(LLVMCodeBuilder &CB, const Var<T> &V) {
  using ResultT = std::remove_reference_t<U>;
  Var<ResultT> Res = declVar<ResultT>(CB, "convert.");
  llvm::Value *Converted = convert<T, U>(CB, V.loadValue());
  Res.storeValue(Converted);
  return Res;
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

  llvm::Value *LHS = convert<T, CommonT>(CB, L.loadValue());
  llvm::Value *RHS = convert<U, CommonT>(CB, R.loadValue());

  llvm::Value *Result = nullptr;
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

  auto &Ctx = LHS.CB.getContext();
  llvm::Type *RHSType = TypeMap<remove_cvref_t<U>>::get(Ctx);

  llvm::Value *RHS = nullptr;
  if constexpr (std::is_integral_v<remove_cvref_t<U>>) {
    RHS = LHS.CB.getConstantInt(RHSType, ConstValue);
  } else {
    RHS = LHS.CB.getConstantFP(RHSType, ConstValue);
  }

  llvm::Value *LHSVal = LHS.loadValue();

  RHS = convert<U, T>(LHS.CB, RHS);
  llvm::Value *Result = nullptr;

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

  llvm::Value *LHS = L.loadValue();
  llvm::Value *RHS = convert<U, T>(CB, R.loadValue());

  llvm::Value *Result = nullptr;
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
template <typename U, typename>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::Var(const Var<U> &V)
    : VarStorageOwner<VarStorage>(V.CB) {
  llvm::Type *TargetTy = TypeMap<remove_cvref_t<T>>::get(CB.getContext());
  Storage = CB.createScalarStorage("conv.var", TargetTy);

  auto *Converted = convert<U, T>(CB, V.loadValue());
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
    llvm::Value *PtrVal = PtrStorage->loadPointer();
    llvm::Type *ElemTy = PtrStorage->getValueType();
    unsigned AddrSpace = CB.getAddressSpaceFromValue(PtrVal);
    llvm::Type *PtrTy = CB.getPointerType(ElemTy, AddrSpace);
    PtrVal = CB.createBitCast(PtrVal, PtrTy);

    std::unique_ptr<PointerStorage> ResultStorage =
        CB.createPointerStorage("addr.ref.tmp", PtrTy, ElemTy);
    CB.createStore(PtrVal, ResultStorage->getSlot());

    return Var<std::add_pointer_t<T>>(std::move(ResultStorage), CB);
  }

  llvm::Value *Slot = getSlot();
  llvm::Type *ElemTy = getAllocatedType();

  unsigned AddrSpace = CB.getAddressSpace(getSlotType());
  llvm::Type *PtrTy = CB.getPointerType(ElemTy, AddrSpace);
  llvm::Value *PtrVal = CB.createBitCast(Slot, PtrTy);

  std::unique_ptr<PointerStorage> ResultStorage =
      CB.createPointerStorage("addr.tmp", PtrTy, ElemTy);
  CB.createStore(PtrVal, ResultStorage->getSlot());

  return Var<std::add_pointer_t<T>>(std::move(ResultStorage), CB);
}

template <typename T>
template <typename U>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator=(
    const Var<U> &V) {
  static_assert(is_mutable_v<T>, "Cannot assign to Var<const T>");
  auto *Converted = convert<U, T>(CB, V.loadValue());
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

  llvm::Type *LHSType = getValueType();

  if (CB.isIntegerTy(LHSType)) {
    storeValue(CB.getConstantInt(LHSType, ConstValue));
  } else if (CB.isFloatingPointTy(LHSType)) {
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
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
        return CB.createAdd(L, R);
      },
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
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
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
        return CB.createSub(L, R);
      },
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
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
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
        return CB.createMul(L, R);
      },
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
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
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
        return CB.createSDiv(L, R);
      },
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
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
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
        return CB.createSRem(L, R);
      },
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
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
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
        return CB.createAdd(L, R);
      },
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
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
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
        return CB.createSub(L, R);
      },
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
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
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
        return CB.createMul(L, R);
      },
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
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
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
        return CB.createSDiv(L, R);
      },
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
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
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
        return CB.createSRem(L, R);
      },
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
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
  llvm::Value *V = loadValue();
  llvm::Value *ResV = nullptr;
  if constexpr (std::is_same_v<remove_cvref_t<T>, bool>) {
    ResV = CB.createNot(V);
  } else if constexpr (std::is_integral_v<remove_cvref_t<T>>) {
    llvm::Value *Zero = CB.getConstantInt(getValueType(), 0);
    ResV = CB.createICmpEQ(V, Zero);
  } else {
    llvm::Value *Zero = CB.getConstantFP(getValueType(), 0.0);
    ResV = CB.createFCmpOEQ(V, Zero);
  }
  auto Ret = declVar<bool>(CB, "not.");
  Ret.storeValue(ResV);
  return Ret;
}

template <typename T>
Var<std::add_lvalue_reference_t<std::remove_extent_t<T>>>
Var<T, std::enable_if_t<std::is_array_v<T>>>::operator[](size_t Index) {
  auto *ArrayTy = getAllocatedType();
  auto *BasePointer = getSlot();

  auto *GEP = CB.createConstInBoundsGEP2_64(ArrayTy, BasePointer, 0, Index);
  llvm::Type *ElemTy = getValueType();
  unsigned AddrSpace = CB.getAddressSpace(getSlotType());
  llvm::Type *ElemPtrTy = CB.getPointerType(ElemTy, AddrSpace);

  std::unique_ptr<PointerStorage> ResultStorage =
      CB.createPointerStorage("elem.ptr", ElemPtrTy, ElemTy);
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
  auto *ArrayTy = getAllocatedType();
  auto *BasePointer = getSlot();

  llvm::Value *IdxVal = Index.loadValue();
  llvm::Value *Zero = CB.getConstantInt(Index.getValueType(), 0);
  auto *GEP = CB.createInBoundsGEP(ArrayTy, BasePointer, {Zero, IdxVal});
  llvm::Type *ElemTy = getValueType();
  unsigned AddrSpace = CB.getAddressSpace(getSlotType());
  llvm::Type *ElemPtrTy = CB.getPointerType(ElemTy, AddrSpace);

  std::unique_ptr<PointerStorage> ResultStorage =
      CB.createPointerStorage("elem.ptr", ElemPtrTy, ElemTy);
  CB.createStore(GEP, ResultStorage->getSlot());

  return Var<std::add_lvalue_reference_t<std::remove_extent_t<T>>>(
      std::move(ResultStorage), CB);
}

template <typename T>
Var<std::add_lvalue_reference_t<
    std::remove_pointer_t<std::remove_reference_t<T>>>>
Var<T, std::enable_if_t<is_pointer_unref_v<T>>>::operator[](size_t Index) {
  auto *PointerElemTy =
      TypeMap<std::remove_pointer_t<std::remove_reference_t<T>>>::get(
          CB.getContext());
  auto *Ptr = loadPointer();
  auto *GEP = CB.createConstInBoundsGEP1_64(PointerElemTy, Ptr, Index);
  unsigned AddrSpace = CB.getAddressSpace(getAllocatedType());
  llvm::Type *ElemPtrTy = CB.getPointerType(PointerElemTy, AddrSpace);

  std::unique_ptr<PointerStorage> ResultStorage =
      CB.createPointerStorage("elem.ptr", ElemPtrTy, PointerElemTy);
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

  auto *PointeeType =
      TypeMap<std::remove_pointer_t<std::remove_reference_t<T>>>::get(
          CB.getContext());
  auto *Ptr = loadPointer();
  auto *IdxValue = Index.loadValue();
  auto *GEP = CB.createInBoundsGEP(PointeeType, Ptr, {IdxValue});
  unsigned AddrSpace = CB.getAddressSpace(getAllocatedType());
  llvm::Type *ElemPtrTy = CB.getPointerType(PointeeType, AddrSpace);

  std::unique_ptr<PointerStorage> ResultStorage =
      CB.createPointerStorage("elem.ptr", ElemPtrTy, PointeeType);
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
  llvm::Value *PtrVal = loadPointer();
  llvm::Type *ElemTy = getValueType();

  unsigned AddrSpace = CB.getAddressSpace(getAllocatedType());
  llvm::Type *PointeePtrTy = CB.getPointerType(ElemTy, AddrSpace);
  llvm::Type *TargetPtrTy = CB.getPointerTypeUnqual(PointeePtrTy);

  PtrVal = CB.createBitCast(PtrVal, PointeePtrTy);

  std::unique_ptr<PointerStorage> ResultStorage =
      CB.createPointerStorage("addr.ptr.tmp", TargetPtrTy, PointeePtrTy);
  CB.createStore(PtrVal, ResultStorage->getSlot());

  return Var<std::add_pointer_t<T>>(std::move(ResultStorage), CB);
}

template <typename T>
template <typename OffsetT>
std::enable_if_t<std::is_arithmetic_v<OffsetT>,
                 Var<T, std::enable_if_t<is_pointer_unref_v<T>>>>
Var<T, std::enable_if_t<is_pointer_unref_v<T>>>::operator+(
    const Var<OffsetT> &Offset) const {
  auto *OffsetVal = Offset.loadValue();
  auto *IdxVal = convert<OffsetT, int64_t>(CB, OffsetVal);

  auto *BasePtr = loadPointer();
  auto *ElemTy = getValueType();

  auto *GEP = CB.createInBoundsGEP(ElemTy, BasePtr, IdxVal, "ptr.add");

  unsigned AddrSpace = CB.getAddressSpace(getAllocatedType());
  auto *ElemPtrTy = CB.getPointerType(ElemTy, AddrSpace);

  std::unique_ptr<PointerStorage> ResultStorage =
      CB.createPointerStorage("ptr.add.tmp", ElemPtrTy, ElemTy);
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
  auto *IntTy = CB.getInt64Ty();
  llvm::Value *IdxVal = CB.getConstantInt(IntTy, Offset);

  auto *BasePtr = loadPointer();
  auto *ElemTy = getValueType();

  auto *GEP = CB.createInBoundsGEP(ElemTy, BasePtr, {IdxVal}, "ptr.add");

  unsigned AddrSpace = CB.getAddressSpace(getAllocatedType());
  auto *ElemPtrTy = CB.getPointerType(ElemTy, AddrSpace);

  std::unique_ptr<PointerStorage> ResultStorage =
      CB.createPointerStorage("ptr.add.tmp", ElemPtrTy, ElemTy);
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
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
        return CB.createICmpSGT(L, R);
      },
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
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
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
        return CB.createICmpSGE(L, R);
      },
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
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
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
        return CB.createICmpSLT(L, R);
      },
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
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
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
        return CB.createICmpSLE(L, R);
      },
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
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
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
        return CB.createICmpEQ(L, R);
      },
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
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
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
        return CB.createICmpNE(L, R);
      },
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
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

  template <typename U> llvm::Value *operator()(const Var<U> &Operand) const {
    return convert<U, T>(CB, Operand.loadValue());
  }
};

template <typename T, typename... Operands>
static Var<T> emitIntrinsic(const std::string &IntrinsicName,
                            llvm::Type *ResultType, const Operands &...Ops) {
  static_assert(sizeof...(Ops) > 0, "Intrinsic requires at least one operand");

  LLVMCodeBuilder &CB = std::get<0>(std::tie(Ops...)).CB;
  auto CheckFn = [&CB](const auto &Operand) {
    if (&Operand.CB != &CB)
      reportFatalError("Variables should belong to the same function");
  };
  (CheckFn(Ops), ...);

  IntrinsicOperandConverter<T> ConvertOperand{CB};

  std::vector<llvm::Type *> ArgTys(sizeof...(Ops), ResultType);
  llvm::Value *Call = CB.createCall(IntrinsicName, ResultType, ArgTys,
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

  auto *ResultType = R.CB.getFloatTy();
  auto RFloat = convertVar<float>(R.CB, R);
  std::string IntrinsicName = "llvm.pow.f32";
#if PROTEUS_ENABLE_CUDA
  if (L.CB.getTargetModel() == TargetModelType::CUDA)
    IntrinsicName = "__nv_powf";
#endif

  return emitIntrinsic<float>(IntrinsicName, ResultType, L, RFloat);
}

template <typename T> Var<float> sqrtf(const Var<T> &R) {
  static_assert(std::is_convertible_v<T, float>,
                "sqrtf requires floating-point type");

  auto *ResultType = R.CB.getFloatTy();
  auto RFloat = convertVar<float>(R.CB, R);
  std::string IntrinsicName = "llvm.sqrt.f32";
#if PROTEUS_ENABLE_CUDA
  if (R.CB.getTargetModel() == TargetModelType::CUDA)
    IntrinsicName = "__nv_sqrtf";
#endif

  return emitIntrinsic<float>(IntrinsicName, ResultType, RFloat);
}

template <typename T> Var<float> expf(const Var<T> &R) {
  static_assert(std::is_convertible_v<T, float>,
                "expf requires floating-point type");

  auto *ResultType = R.CB.getFloatTy();
  auto RFloat = convertVar<float>(R.CB, R);
  std::string IntrinsicName = "llvm.exp.f32";
#if PROTEUS_ENABLE_CUDA
  if (R.CB.getTargetModel() == TargetModelType::CUDA)
    IntrinsicName = "__nv_expf";
#endif

  return emitIntrinsic<float>(IntrinsicName, ResultType, RFloat);
}

template <typename T> Var<float> sinf(const Var<T> &R) {
  static_assert(std::is_convertible_v<T, float>,
                "sinf requires floating-point type");

  auto *ResultType = R.CB.getFloatTy();
  auto RFloat = convertVar<float>(R.CB, R);
  std::string IntrinsicName = "llvm.sin.f32";
#if PROTEUS_ENABLE_CUDA
  if (R.CB.getTargetModel() == TargetModelType::CUDA)
    IntrinsicName = "__nv_sinf";
#endif

  return emitIntrinsic<float>(IntrinsicName, ResultType, RFloat);
}

template <typename T> Var<float> cosf(const Var<T> &R) {
  static_assert(std::is_convertible_v<T, float>,
                "cosf requires floating-point type");

  auto *ResultType = R.CB.getFloatTy();
  auto RFloat = convertVar<float>(R.CB, R);
  std::string IntrinsicName = "llvm.cos.f32";
#if PROTEUS_ENABLE_CUDA
  if (R.CB.getTargetModel() == TargetModelType::CUDA)
    IntrinsicName = "__nv_cosf";
#endif

  return emitIntrinsic<float>(IntrinsicName, ResultType, RFloat);
}

template <typename T> Var<float> fabs(const Var<T> &R) {
  static_assert(std::is_convertible_v<T, float>,
                "fabs requires floating-point type");

  auto *ResultType = R.CB.getFloatTy();
  auto RFloat = convertVar<float>(R.CB, R);
  std::string IntrinsicName = "llvm.fabs.f32";
#if PROTEUS_ENABLE_CUDA
  if (R.CB.getTargetModel() == TargetModelType::CUDA)
    IntrinsicName = "__nv_fabsf";
#endif

  return emitIntrinsic<float>(IntrinsicName, ResultType, RFloat);
}

template <typename T> Var<float> truncf(const Var<T> &R) {
  static_assert(std::is_convertible_v<T, float>,
                "truncf requires floating-point type");

  auto *ResultType = R.CB.getFloatTy();
  auto RFloat = convertVar<float>(R.CB, R);
  std::string IntrinsicName = "llvm.trunc.f32";
#if PROTEUS_ENABLE_CUDA
  if (R.CB.getTargetModel() == TargetModelType::CUDA)
    IntrinsicName = "__nv_truncf";
#endif

  return emitIntrinsic<float>(IntrinsicName, ResultType, RFloat);
}

template <typename T> Var<float> logf(const Var<T> &R) {
  static_assert(std::is_convertible_v<T, float>,
                "logf requires floating-point type");

  auto *ResultType = R.CB.getFloatTy();
  auto RFloat = convertVar<float>(R.CB, R);
  std::string IntrinsicName = "llvm.log.f32";
#if PROTEUS_ENABLE_CUDA
  if (R.CB.getTargetModel() == TargetModelType::CUDA)
    IntrinsicName = "__nv_logf";
#endif

  return emitIntrinsic<float>(IntrinsicName, ResultType, RFloat);
}

template <typename T> Var<float> absf(const Var<T> &R) {
  static_assert(std::is_convertible_v<T, float>,
                "absf requires floating-point type");

  auto *ResultType = R.CB.getFloatTy();
  auto RFloat = convertVar<float>(R.CB, R);
  std::string IntrinsicName = "llvm.fabs.f32";
#if PROTEUS_ENABLE_CUDA
  if (R.CB.getTargetModel() == TargetModelType::CUDA)
    IntrinsicName = "__nv_fabsf";
#endif

  return emitIntrinsic<float>(IntrinsicName, ResultType, RFloat);
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
