#include "proteus/JitFrontend.hpp"

namespace proteus {

template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
struct IntegerType {};

Var::Var(AllocaInst *Alloca, Func &Fn, Type *PointerElemType)
    : Alloca(Alloca), Fn(Fn), PointerElemType(PointerElemType) {}

Value *Var::getValue() const {
  auto &IRB = Fn.getIRB();
  Type *AllocaType = Alloca->getAllocatedType();
  if (AllocaType->isPointerTy()) {
    auto *Ptr = IRB.CreateLoad(AllocaType, Alloca);
    return IRB.CreateLoad(PointerElemType, Ptr);
  }
  return IRB.CreateLoad(AllocaType, Alloca);
}

Type *Var::getValueType() const {
  Type *AllocaType = Alloca->getAllocatedType();
  if (AllocaType->isPointerTy()) {
    return PointerElemType;
  }
  return AllocaType;
}

bool Var::isPointer() const {
  Type *AllocaType = Alloca->getAllocatedType();
  if (AllocaType->isPointerTy()) {
    assert(PointerElemType && "Expected pointer type");
    return true;
  }
  return false;
}

void Var::storeValue(Value *Val) {
  auto &IRB = Fn.getIRB();
  Type *AllocaType = Alloca->getAllocatedType();
  if (AllocaType->isPointerTy()) {
    auto *Ptr = IRB.CreateLoad(AllocaType, Alloca);
    IRB.CreateStore(Val, Ptr);
  } else {
    IRB.CreateStore(Val, Alloca);
  }
}

void Var::storePointer(Value *Ptr) {
  auto &IRB = Fn.getIRB();
  assert(isPointer() && "Expected pointer type");
  IRB.CreateStore(Ptr, Alloca);
}

Var &Var::operator+(Var &Other) {
  auto &IRB = Fn.getIRB();
  Type *LHSType = getValueType();
  Type *RHSType = Other.getValueType();
  assert(LHSType == RHSType && "Expected matching types");

  auto *ResultType = LHSType;
  Var &ResultVar = Fn.declVarInternal("res.", ResultType);

  Value *LHS = getValue();
  Value *RHS = Other.getValue();

  if (ResultType->isIntegerTy()) {
    auto *Result = IRB.CreateAdd(LHS, RHS);
    ResultVar.storeValue(Result);
    return ResultVar;
  }

  if (ResultType->isFloatingPointTy()) {
    auto *Result = IRB.CreateFAdd(LHS, RHS);
    ResultVar.storeValue(Result);
    return ResultVar;
  }

  PROTEUS_FATAL_ERROR("Unsupported type");
}

template <typename T> Var &Var::operator+(const T &ConstValue) {
  auto &IRB = Fn.getIRB();
  Type *LHSType = getValueType();

  auto *ResultType = LHSType;
  Var &ResultVar = Fn.declVarInternal("res.", ResultType);

  Value *LHS = getValue();
  auto *RHS = ConstantFP::get(ResultType, ConstValue);
  auto *Result = IRB.CreateFAdd(LHS, RHS);

  ResultVar.storeValue(Result);

  return ResultVar;
}

Var &Var::operator=(const Var &Other) {
  auto &IRB = Fn.getIRB();
  Type *LHSType = getValueType();
  Type *RHSType = Other.getValueType();
  assert(LHSType == RHSType && "Expected matching types");

  Value *RHS = Other.getValue();
  storeValue(RHS);

  return *this;
}

template <typename T> Var &Var::operator=(const T &ConstValue) {
  Function *F = Fn.getFunction();
  auto &IRB = Fn.getIRB();

  Type *LHSType = getValueType();

  if (LHSType->isIntegerTy()) {
    storeValue(ConstantInt::get(LHSType, ConstValue));
  } else if (LHSType->isFloatingPointTy()) {
    storeValue(ConstantFP::get(LHSType, ConstValue));
  } else {
    PROTEUS_FATAL_ERROR("Unsupported type");
  }

  return *this;
}

Var &Var::operator>(const double &ConstValue) {
  Function *F = Fn.getFunction();
  auto &IRB = Fn.getIRB();

  Type *BoolType = Type::getInt1Ty(F->getContext());
  Var &ResultVar = Fn.declVarInternal("res.", BoolType);

  auto *ConstType = TypeMap<double>::get(F->getContext());
  auto *LHS = getValue();
  auto *Result = IRB.CreateFCmpOGT(LHS, ConstantFP::get(ConstType, ConstValue));
  ResultVar.storeValue(Result);

  return ResultVar;
}

Var &Var::operator<(const double &ConstValue) {
  Function *F = Fn.getFunction();
  auto &IRB = Fn.getIRB();

  Type *BoolType = Type::getInt1Ty(F->getContext());
  Var &ResultVar = Fn.declVarInternal("res.", BoolType);
  auto *ResultAlloca = Fn.emitAlloca(Type::getInt1Ty(F->getContext()), "res.");

  auto *ConstType = TypeMap<double>::get(F->getContext());
  auto *LHS = getValue();
  auto *Result = IRB.CreateFCmpOLT(LHS, ConstantFP::get(ConstType, ConstValue));

  ResultVar.storeValue(Result);

  return ResultVar;
}

Var &Var::operator<(const Var &Other) {
  Function *F = Fn.getFunction();
  Type *LHSType = getValueType();
  Type *RHSType = Other.getValueType();
  if (LHSType != RHSType) {
    dbgs() << "Function: " << *F << "\n";
    dbgs() << "LHSType: " << *LHSType << "\n";
    dbgs() << "RHSType: " << *RHSType << "\n";
    PROTEUS_FATAL_ERROR("Type mismatch");
  }
  auto &IRB = Fn.getIRB();

  Type *BoolType = Type::getInt1Ty(F->getContext());
  Var &ResultVar = Fn.declVarInternal("res.", BoolType);

  auto *LHS = getValue();
  auto *RHS = Other.getValue();

  if (LHSType->isIntegerTy()) {
    auto *Result = IRB.CreateICmpSLT(LHS, RHS);
    ResultVar.storeValue(Result);
    return ResultVar;
  }

  if (LHSType->isFloatingPointTy()) {
    auto *Result = IRB.CreateFCmpOLT(LHS, RHS);
    ResultVar.storeValue(Result);
    return ResultVar;
  }

  PROTEUS_FATAL_ERROR("Unsupported type");
}

Var &Var::operator>(const Var &Other) {
  Function *F = Fn.getFunction();
  Type *LHSType = getValueType();
  Type *RHSType = Other.getValueType();
  assert(LHSType == RHSType && "Expected matching types");
  auto &IRB = Fn.getIRB();

  Type *BoolType = Type::getInt1Ty(F->getContext());
  Var &ResultVar = Fn.declVarInternal("res.", BoolType);

  auto *LHS = getValue();
  auto *RHS = Other.getValue();
  auto *Result = IRB.CreateFCmpOGT(LHS, RHS);

  ResultVar.storeValue(Result);

  return ResultVar;
}

Var &Var::operator[](size_t I) {
  auto &IRB = Fn.getIRB();

  assert(isPointer() && "Expected pointer type");

  auto &ResultVar = Fn.declVarInternal("res.", PointerElemType->getPointerTo(),
                                       PointerElemType);
  auto *Ptr = IRB.CreateLoad(Alloca->getAllocatedType(), Alloca);
  auto *GEP = IRB.CreateConstInBoundsGEP1_64(PointerElemType, Ptr, I);

  ResultVar.storePointer(GEP);

  return ResultVar;
}

Var &Var::operator[](const Var &IdxVar) {
  auto &IRB = Fn.getIRB();

  assert(isPointer() && "Expected pointer type");

  auto &ResultVar = Fn.declVarInternal("res.", PointerElemType->getPointerTo(),
                                       PointerElemType);
  auto *Ptr = IRB.CreateLoad(Alloca->getAllocatedType(), Alloca);
  Value *Idx = IdxVar.getValue();
  auto *GEP = IRB.CreateInBoundsGEP(PointerElemType, Ptr, {Idx});

  ResultVar.storePointer(GEP);

  return ResultVar;
}

// Integral types
template Var &Var::operator=<int>(const int &);
template Var &Var::operator=<size_t>(const size_t &);

// Floating point types
template Var &Var::operator=<double>(const double &);
template Var &Var::operator+<double>(const double &);

} // namespace proteus