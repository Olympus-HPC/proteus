#ifndef PROTEUS_FRONTEND_VAR_HPP
#define PROTEUS_FRONTEND_VAR_HPP

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

namespace proteus {

class Func;

using namespace llvm;

struct Var {
  AllocaInst *Alloca;
  Type *PointerElemType;
  Func &Fn;

  Var(AllocaInst *Alloca, Func &Fn, Type *PointerElemType = nullptr);

  Var &operator+(Var &Other);
  template <typename T> Var &operator+(const T &ConstValue);

  Var &operator>(const double &ConstValue);
  Var &operator<(const double &ConstValue);
  Var &operator<(const Var &Other);
  Var &operator>(const Var &Other);

  Var &operator=(const Var &Other);
  template <typename T> Var &operator=(const T &ConstValue);

  Var &operator[](size_t I);
  Var &operator[](const Var &I);

  Value *getValue() const;
  Type *getValueType() const;
  void storeValue(Value *Val);
  void storePointer(Value *Ptr);

  bool isPointer() const;
};

} // namespace proteus

#endif // PROTEUS_FRONTEND_VAR_HPP
