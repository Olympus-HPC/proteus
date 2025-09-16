#ifndef PROTEUS_FRONTEND_VAR_HPP
#define PROTEUS_FRONTEND_VAR_HPP

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

namespace proteus {

class FuncBase;

using namespace llvm;

enum class VarKind { Invalid, Scalar, Pointer, Array };

template <typename T> struct VarT {};

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

} // namespace proteus

#endif // PROTEUS_FRONTEND_VAR_HPP
