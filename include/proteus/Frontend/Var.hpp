#ifndef PROTEUS_FRONTEND_VAR_HPP
#define PROTEUS_FRONTEND_VAR_HPP

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <variant>

namespace proteus {

class FuncBase;

using namespace llvm;

template <typename T> struct VarT {};

struct AllocaStorage {
  AllocaInst *Alloca;
  Type *PointerElemType;

  Value *getValue(IRBuilderBase &IRB) const;
  Type *getValueType() const;
  StringRef getName() const;
  void storeValue(IRBuilderBase &IRB, Value *Val);
  void storePointer(IRBuilderBase &IRB, Value *Ptr);
  bool isPointer() const;
};

struct BorrowedStorage {
  Value *PointerValue;
  Type *PointerElemType;

  Value *getValue(IRBuilderBase &IRB) const;
  Type *getValueType() const;
  StringRef getName() const;
  void storeValue(IRBuilderBase &IRB, Value *Val);
  void storePointer(IRBuilderBase &IRB, Value *Ptr);
  bool isPointer() const;
};

using VarStorage = std::variant<AllocaStorage, BorrowedStorage>;

struct Var {
  VarStorage Storage;
  FuncBase &Fn;

  Var(AllocaInst *Alloca, FuncBase &Fn, Type *PointerElemType = nullptr);
  Var(Value *PointerValue, FuncBase &Fn, Type *PointerElemType);

  static Var fromBorrowed(Value *PointerValue, FuncBase &Fn, Type *PointerElemType);

  StringRef getName();

  Value *getValue() const;
  Type *getValueType() const;
  void storeValue(Value *Val);
  void storePointer(Value *Ptr);

  bool isPointer() const;

  AllocaInst *getAlloca() const;
  Type *getPointerElemType() const;

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

// Declare usual arithmetic conversion helper functions.
Value *convert(IRBuilderBase IRB, Value *V, Type *TargetType);
Type *getCommonType(const DataLayout &DL, Type *T1, Type *T2);

// Declare intrinsic math functions.
Var &powf(const Var &L, const Var &R);
Var &sqrtf(const Var &R);
Var &min(const Var &L, const Var &R);

} // namespace proteus

#endif // PROTEUS_FRONTEND_VAR_HPP
