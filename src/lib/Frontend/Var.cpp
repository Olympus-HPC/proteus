#include "proteus/JitFrontend.hpp"

namespace proteus {

Var::Var(AllocaInst *Alloca, Func &Fn) : Alloca(Alloca), Fn(Fn) {}

Var &Var::operator+(Var &Other) {
  Function *F = Fn.getFunction();
  auto &IRB = Fn.IRB;
  auto AllocaIP = IRBuilderBase::InsertPoint(&F->getEntryBlock(),
                                             F->getEntryBlock().begin());
  IRB.restoreIP(AllocaIP);
  auto *ResultAlloca =
      IRB.CreateAlloca(Alloca->getAllocatedType(), nullptr, "res");

  IRB.restoreIP(Fn.IP);
  auto *LHS = IRB.CreateLoad(Alloca->getAllocatedType(), Alloca);
  auto *RHS = IRB.CreateLoad(Other.Alloca->getAllocatedType(), Other.Alloca);
  auto *Result = IRB.CreateFAdd(LHS, RHS);
  IRB.CreateStore(Result, ResultAlloca);

  Fn.Variables.emplace_back(ResultAlloca, Fn);
  return Fn.Variables.back();
}

Var &Var::operator+(const double &ConstValue) {
  Function *F = Fn.getFunction();
  auto &IRB = Fn.IRB;
  auto AllocaIP = IRBuilderBase::InsertPoint(&F->getEntryBlock(),
                                             F->getEntryBlock().begin());
  IRB.restoreIP(AllocaIP);
  auto *ResultAlloca =
      IRB.CreateAlloca(Alloca->getAllocatedType(), nullptr, "res");

  IRB.restoreIP(Fn.IP);
  auto *LHS = IRB.CreateLoad(Alloca->getAllocatedType(), Alloca);
  auto *RHS = ConstantFP::get(Alloca->getAllocatedType(), ConstValue);
  auto *Result = IRB.CreateFAdd(LHS, RHS);
  IRB.CreateStore(Result, ResultAlloca);

  Fn.Variables.emplace_back(ResultAlloca, Fn);
  return Fn.Variables.back();
}

Var &Var::operator=(const Var &Other) {
  Function *F = Fn.getFunction();
  auto &IRB = Fn.IRB;

  IRB.restoreIP(Fn.IP);
  auto *RHS = IRB.CreateLoad(Other.Alloca->getAllocatedType(), Other.Alloca);
  IRB.CreateStore(RHS, Alloca);
  return *this;
}

Var &Var::operator=(const double &ConstValue) {
  Function *F = Fn.getFunction();
  auto &IRB = Fn.IRB;

  IRB.restoreIP(Fn.IP);
  IRB.CreateStore(ConstantFP::get(Alloca->getAllocatedType(), ConstValue),
                  Alloca);
  return *this;
}

Var &Var::operator>(const double &ConstValue) {
  Function *F = Fn.getFunction();
  auto &IRB = Fn.IRB;

  auto AllocaIP = IRBuilderBase::InsertPoint(&F->getEntryBlock(),
                                             F->getEntryBlock().begin());
  IRB.restoreIP(AllocaIP);
  auto *ResultAlloca =
      IRB.CreateAlloca(Type::getInt1Ty(F->getContext()), nullptr, "res");

  IRB.restoreIP(Fn.IP);
  auto *LHS = IRB.CreateLoad(Alloca->getAllocatedType(), Alloca);
  auto *Result = IRB.CreateFCmpOGT(
      LHS, ConstantFP::get(Alloca->getAllocatedType(), ConstValue));
  IRB.CreateStore(Result, ResultAlloca);

  Fn.Variables.emplace_back(ResultAlloca, Fn);
  return Fn.Variables.back();
}

} // namespace proteus