#ifndef PROTEUS_FRONTEND_VAR_HPP
#define PROTEUS_FRONTEND_VAR_HPP

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <type_traits>

namespace proteus {

class FuncBase;

using namespace llvm;

enum class VarKind { Invalid, Scalar, Pointer, Array };

template <typename T> struct VarT {};



class VarStorage;
class PointerStorage;
class ArrayStorage;

struct Var {
  AllocaInst *Alloca;
  FuncBase &Fn;

  VarKind Kind = VarKind::Invalid;

  virtual ~Var() = default;

  Var(AllocaInst *Alloca, FuncBase &Fn);
  Var(FuncBase &Fn);

  // Disable copying/moving to prevent object slicing and enforce reference
  // semantics.
  Var(const Var &) = delete;
  Var(Var &&) = delete;

  virtual StringRef getName() const = 0;

  // Value accessors
  virtual Value *getValue() const = 0;
  virtual Type *getValueType() const = 0;

  virtual void storeValue(Value *Val) = 0;

  // Pointer-only hooks
  virtual Value *getPointerValue() const = 0;
  virtual void storePointer(Value *Ptr) = 0;

  virtual AllocaInst *getAlloca() const;

  virtual VarKind kind() const;

  virtual Var &index(size_t I) = 0;
  virtual Var &index(const Var &I) = 0;

  // Declare member Operators.
  Var &operator+(const Var &Other) const;
  Var &operator-(const Var &Other) const;
  Var &operator*(const Var &Other) const;
  Var &operator/(const Var &Other) const;
  Var &operator%(const Var &Other) const;

  template <typename T>
  std::enable_if_t<std::is_arithmetic_v<T>, Var &>
  operator+(const T &ConstValue) const;

  template <typename T>
  std::enable_if_t<std::is_arithmetic_v<T>, Var &>
  operator-(const T &ConstValue) const;

  template <typename T>
  std::enable_if_t<std::is_arithmetic_v<T>, Var &>
  operator*(const T &ConstValue) const;

  template <typename T>
  std::enable_if_t<std::is_arithmetic_v<T>, Var &>
  operator/(const T &ConstValue) const;

  template <typename T>
  std::enable_if_t<std::is_arithmetic_v<T>, Var &>
  operator%(const T &ConstValue) const;

  Var &operator+=(Var &Other);
  Var &operator-=(Var &Other);
  Var &operator*=(Var &Other);
  Var &operator/=(Var &Other);
  Var &operator%=(Var &Other);

  template <typename T>
  std::enable_if_t<std::is_arithmetic_v<T>, Var &>
  operator+=(const T &ConstValue);

  template <typename T>
  std::enable_if_t<std::is_arithmetic_v<T>, Var &>
  operator-=(const T &ConstValue);

  template <typename T>
  std::enable_if_t<std::is_arithmetic_v<T>, Var &>
  operator*=(const T &ConstValue);

  template <typename T>
  std::enable_if_t<std::is_arithmetic_v<T>, Var &>
  operator/=(const T &ConstValue);

  template <typename T>
  std::enable_if_t<std::is_arithmetic_v<T>, Var &>
  operator%=(const T &ConstValue);

  Var &operator>(const Var &Other) const;
  Var &operator<(const Var &Other) const;
  Var &operator>=(const Var &Other) const;
  Var &operator<=(const Var &Other) const;
  Var &operator==(const Var &Other) const;
  Var &operator!=(const Var &Other) const;

  template <typename T>
  std::enable_if_t<std::is_arithmetic_v<T>, Var &>
  operator>(const T &ConstValue) const;

  template <typename T>
  std::enable_if_t<std::is_arithmetic_v<T>, Var &>
  operator>=(const T &ConstValue) const;

  template <typename T>
  std::enable_if_t<std::is_arithmetic_v<T>, Var &>
  operator<(const T &ConstValue) const;

  template <typename T>
  std::enable_if_t<std::is_arithmetic_v<T>, Var &>
  operator<=(const T &ConstValue) const;

  template <typename T>
  std::enable_if_t<std::is_arithmetic_v<T>, Var &>
  operator==(const T &ConstValue) const;

  template <typename T>
  std::enable_if_t<std::is_arithmetic_v<T>, Var &>
  operator!=(const T &ConstValue) const;

  Var &operator=(const Var &Other);

  template <typename T, typename = std::enable_if<std::is_arithmetic_v<T>>>
  Var &operator=(const T &ConstValue);

  Var &operator[](size_t I);
  Var &operator[](const Var &I);
};

// Declare non-member operators.
template <typename T>
std::enable_if_t<std::is_arithmetic_v<T>, Var &> operator+(const T &ConstValue,
                                                           const Var &Other);
template <typename T>
std::enable_if_t<std::is_arithmetic_v<T>, Var &> operator-(const T &ConstValue,
                                                           const Var &Other);
template <typename T>
std::enable_if_t<std::is_arithmetic_v<T>, Var &> operator*(const T &ConstValue,
                                                           const Var &Other);
template <typename T>
std::enable_if_t<std::is_arithmetic_v<T>, Var &> operator/(const T &ConstValue,
                                                           const Var &Other);
template <typename T>
std::enable_if_t<std::is_arithmetic_v<T>, Var &> operator%(const T &ConstValue,
                                                           const Var &Other);

// Declare usual arithmetic conversion helper functions.
Value *convert(IRBuilderBase IRB, Value *V, Type *TargetType);
Type *getCommonType(const DataLayout &DL, Type *T1, Type *T2);

// Declare intrinsic math functions.
Var &powf(const Var &L, const Var &R);
Var &sqrtf(const Var &R);
Var &min(const Var &L, const Var &R);

struct ScalarVar final : Var {
  // ScalarVar: wraps an alloca of a scalar value.
  // Backing slot for the scalar value.
  AllocaInst *Slot = nullptr;

  explicit ScalarVar(AllocaInst *Slot, FuncBase &Fn);

  StringRef getName() const override;
  Type *getValueType() const override;
  Value *getValue() const override;
  void storeValue(Value *Val) override;
  // Scalar does not support pointer semantics or indexing
  Value *getPointerValue() const override;
  void storePointer(Value *Ptr) override;
  Var &index(size_t I) override;
  Var &index(const Var &I) override;
  VarKind kind() const override;
  AllocaInst *getAlloca() const override;
};
struct PointerVar final : Var {
  Type *PointerElemTy = nullptr;

  explicit PointerVar(AllocaInst *PtrSlot, FuncBase &Fn, Type *ElemTy);

  StringRef getName() const override;
  Type *getValueType() const override;
  Value *getValue() const override;
  void storeValue(Value *Val) override;

  Value *getPointerValue() const override;
  void storePointer(Value *Ptr) override;

  VarKind kind() const override;
  AllocaInst *getAlloca() const override;

  Var &index(size_t I) override;
  Var &index(const Var &I) override;
};
struct ArrayVar final : Var {
  // Holds a pointer to an array aggregate and its type.
  Value *BasePointer = nullptr;
  ArrayType *ArrayTy = nullptr;

  explicit ArrayVar(Value *BasePointer, FuncBase &Fn, ArrayType *ArrayTy);

  StringRef getName() const override;
  Type *getValueType() const override;
  Value *getValue() const override;
  void storeValue(Value *Val) override;
  Value *getPointerValue() const override;
  void storePointer(Value *Ptr) override;
  VarKind kind() const override;

  Var &index(size_t I) override;
  Var &index(const Var &I) override;
};

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
  std::unique_ptr<VarStorage> Storage;
  VarTT(std::unique_ptr<VarStorage> Storage, FuncBase &Fn)
    : Fn(Fn), Storage(std::move(Storage)) {
  }

  // Conversion constructor
  // TODO: Add an is_convertible check.
  template<typename U>
  VarTT(VarTT<U> &Var)
    : Fn(Var.Fn) {
      *this = Var;
  }
  
  VarTT(VarTT &&) = default;
  VarTT &operator=(VarTT &&) = default;
  
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
  std::unique_ptr<ArrayStorage> Storage;

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
  std::unique_ptr<PointerStorage> Storage;

  VarTT(std::unique_ptr<PointerStorage> Storage, FuncBase &Fn)
    : Fn(Fn), Storage(std::move(Storage)) {}
  
  VarTT<ElemType> operator[](size_t Index);

  template <typename IdxT>
  std::enable_if_t<std::is_arithmetic_v<IdxT>, VarTT<ElemType>>
  operator[](const VarTT<IdxT> &Index);

  VarTT<ElemType> operator*();
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
