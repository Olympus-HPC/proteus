#include "proteus/JitFrontend.hpp"

namespace proteus {

Var::Var(AllocaInst *Alloca, Func &Fn, Type *PointerElemType)
    : Alloca(Alloca), Fn(Fn), PointerElemType(PointerElemType) {}

Var &Var::operator+(Var &Other) {
  auto &IRB = Fn.IRB;
  Type *LHSType = Alloca->getAllocatedType();
  Type *RHSType = Other.Alloca->getAllocatedType();
  assert(LHSType == RHSType && "Expected matching types");

  auto *ResultAlloca = Fn.emitAlloca(LHSType, "res.");

  auto *LHS = IRB.CreateLoad(LHSType, Alloca);
  auto *RHS = IRB.CreateLoad(RHSType, Other.Alloca);
  auto *Result = IRB.CreateFAdd(LHS, RHS);
  IRB.CreateStore(Result, ResultAlloca);

  auto &ResultVar = Fn.Variables.emplace_back(ResultAlloca, Fn);
  return ResultVar;
}

Var &Var::operator+(const double &ConstValue) {
  auto &IRB = Fn.IRB;
  Type *LHSType = Alloca->getAllocatedType();

  auto *ResultAlloca = Fn.emitAlloca(LHSType, "res.");

  auto *LHS = IRB.CreateLoad(LHSType, Alloca);
  auto *RHS = ConstantFP::get(LHSType, ConstValue);
  auto *Result = IRB.CreateFAdd(LHS, RHS);
  IRB.CreateStore(Result, ResultAlloca);

  auto &ResultVar = Fn.Variables.emplace_back(ResultAlloca, Fn);
  return ResultVar;
}

Var &Var::operator=(const Var &Other) {
  auto &IRB = Fn.IRB;
  Type *LHSType = Alloca->getAllocatedType();
  Type *RHSType = Other.Alloca->getAllocatedType();
  assert(LHSType == RHSType && "Expected matching types");

  auto *RHS = IRB.CreateLoad(RHSType, Other.Alloca);
  IRB.CreateStore(RHS, Alloca);
  return *this;
}

Var &Var::operator=(const double &ConstValue) {
  Function *F = Fn.getFunction();
  auto &IRB = Fn.IRB;

  IRB.CreateStore(ConstantFP::get(Alloca->getAllocatedType(), ConstValue),
                  Alloca);
  return *this;
}

Var &Var::operator>(const double &ConstValue) {
  Function *F = Fn.getFunction();
  auto &IRB = Fn.IRB;

  auto *ResultAlloca = Fn.emitAlloca(Type::getInt1Ty(F->getContext()), ".res");

  auto *LHS = IRB.CreateLoad(Alloca->getAllocatedType(), Alloca);
  auto *Result = IRB.CreateFCmpOGT(
      LHS, ConstantFP::get(Alloca->getAllocatedType(), ConstValue));
  IRB.CreateStore(Result, ResultAlloca);

  Fn.Variables.emplace_back(ResultAlloca, Fn);
  return Fn.Variables.back();
}

Var &Var::operator[](size_t I) {
  auto &IRB = Fn.IRB;

  auto *ResultAlloca = Fn.emitAlloca(PointerElemType, "res.");

  auto *Ptr = IRB.CreateLoad(Alloca->getAllocatedType(), Alloca);
  auto *GEP = IRB.CreateConstInBoundsGEP1_64(PointerElemType, Ptr, I);
  auto *Load = IRB.CreateLoad(PointerElemType, GEP);
  IRB.CreateStore(Load, ResultAlloca);

  auto &ResultVar = Fn.Variables.emplace_back(ResultAlloca, Fn);
  return ResultVar;
}

} // namespace proteus