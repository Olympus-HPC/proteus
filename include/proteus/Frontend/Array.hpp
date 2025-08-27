#ifndef PROTEUS_FRONTEND_ARRAY_HPP
#define PROTEUS_FRONTEND_ARRAY_HPP

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

#include "proteus/Frontend/Var.hpp"
#include "proteus/AddressSpace.hpp"

namespace proteus {

class FuncBase;

using namespace llvm;

struct Array {
  Value *BasePointer;
  FuncBase &Fn;
  Type *ArrayType;

  AddressSpace AT;

  Array(Value *BasePointer, FuncBase &Fn, Type *ArrayType, AddressSpace AT);
  Var operator[](size_t Index);
  Var operator[](const Var &Index);

};

} // namespace proteus

#endif // PROTEUS_FRONTEND_ARRAY_HPP