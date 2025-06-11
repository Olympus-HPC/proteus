#ifndef PROTEUS_FRONTEND_VAR_HPP
#define PROTEUS_FRONTEND_VAR_HPP

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

namespace proteus {

class Func;

using namespace llvm;

struct Var {
  AllocaInst *Alloca;
  Func &Fn;
  Type *PointerElemType;

  Var(AllocaInst *Alloca, Func &Fn, Type *PointerElemType = nullptr);

  StringRef getName();

  Value *getValue() const;
  Type *getValueType() const;
  void storeValue(Value *Val);
  void storePointer(Value *Ptr);

  bool isPointer() const;

  // Declare member Operators.

  Var &operator+(const Var &Other) const;
  Var &operator-(const Var &Other) const;
  Var &operator*(const Var &Other) const;
  Var &operator/(const Var &Other) const;

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

  Var &operator+=(Var &Other);
  Var &operator-=(Var &Other);
  Var &operator*=(Var &Other);
  Var &operator/=(Var &Other);

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

  template <typename T>
  std::enable_if_t<std::is_arithmetic_v<T>, Var &>
  operator=(const T &ConstValue);

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

// Declare usual arithmetic conversion helper functions.
Value *convert(IRBuilderBase IRB, Value *V, Type *TargetType);
Type *getCommonType(const DataLayout &DL, Type *T1, Type *T2);

// Declare intrinsic math functions.
Var &powf(const Var &L, const Var &R);
Var &sqrtf(const Var &R);

} // namespace proteus

#endif // PROTEUS_FRONTEND_VAR_HPP
