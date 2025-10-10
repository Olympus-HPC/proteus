#include "proteus/Frontend/Var.hpp"
#include "proteus/Error.h"
#include "proteus/Frontend/Func.hpp"
#include "proteus/Frontend/TypeMap.hpp"

namespace proteus {

template <typename IntOp, typename FPOp>
static Var &binOp(const Var &L, const Var &R, IntOp IOp, FPOp FOp) {
  FuncBase &Fn = L.Fn;
  if (&Fn != &R.Fn)
    PROTEUS_FATAL_ERROR("Variables should belong to the same function");
  Function *F = Fn.getFunction();

  auto &DL = F->getParent()->getDataLayout();
  auto &IRB = Fn.getIRBuilder();
  Type *LHSType = L.getValueType();
  Type *RHSType = R.getValueType();

  auto *CommonType = getCommonType(DL, LHSType, RHSType);
  Var &ResultVar = Fn.declVarInternal("res.", CommonType);

  Value *LHS = convert(IRB, L.getValue(), CommonType);
  Value *RHS = convert(IRB, R.getValue(), CommonType);
  Value *Result = nullptr;

  if (CommonType->isIntegerTy()) {
    Result = IOp(IRB, LHS, RHS);
  } else if (CommonType->isFloatingPointTy()) {
    Result = FOp(IRB, LHS, RHS);
  } else
    PROTEUS_FATAL_ERROR("Unsupported type");

  ResultVar.storeValue(Result);

  return ResultVar;
}

template <typename IntOp, typename FPOp>
static Var &cmpOp(const Var &L, const Var &R, IntOp IOp, FPOp FOp) {
  FuncBase &Fn = L.Fn;
  if (&Fn != &R.Fn)
    PROTEUS_FATAL_ERROR("Variables should belong to the same function");
  Function *F = Fn.getFunction();

  auto &DL = F->getParent()->getDataLayout();
  auto &IRB = Fn.getIRBuilder();
  Type *LHSType = L.getValueType();
  Type *RHSType = R.getValueType();

  auto *CommonType = getCommonType(DL, LHSType, RHSType);

  Type *BoolType = Type::getInt1Ty(F->getContext());
  Var &ResultVar = Fn.declVarInternal("res.", BoolType);

  Value *LHS = convert(IRB, L.getValue(), CommonType);
  Value *RHS = convert(IRB, R.getValue(), CommonType);
  Value *Result = nullptr;

  if (CommonType->isIntegerTy()) {
    Result = IOp(IRB, LHS, RHS);
  } else if (CommonType->isFloatingPointTy()) {
    Result = FOp(IRB, LHS, RHS);
  } else
    PROTEUS_FATAL_ERROR("Unsupported type");

  ResultVar.storeValue(Result);

  return ResultVar;
}

Var::Var(AllocaInst *Alloca, FuncBase &Fn) : Alloca(Alloca), Fn(Fn) {}

Var::Var(FuncBase &Fn) : Alloca(nullptr), Fn(Fn) {}

AllocaInst *Var::getAlloca() const { return Alloca; }

Var &Var::operator+(const Var &Other) const {
  return binOp(
      *this, Other,
      [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateAdd(L, R); },
      [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateFAdd(L, R); });
}

Var &Var::operator-(const Var &Other) const {
  return binOp(
      *this, Other,
      [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateSub(L, R); },
      [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateFSub(L, R); });
}

Var &Var::operator*(const Var &Other) const {
  return binOp(
      *this, Other,
      [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateMul(L, R); },
      [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateFMul(L, R); });
}

Var &Var::operator/(const Var &Other) const {
  return binOp(
      *this, Other,
      [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateSDiv(L, R); },
      [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateFDiv(L, R); });
}

Var &Var::operator%(const Var &Other) const {
  return binOp(
      *this, Other,
      [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateSRem(L, R); },
      [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateFRem(L, R); });
}

Var &Var::operator+=(Var &Other) {
  Var &Res = *this + Other;
  auto &IRB = Fn.getIRBuilder();
  this->storeValue(convert(IRB, Res.getValue(), getValueType()));

  return *this;
}

template <typename T>
std::enable_if_t<std::is_arithmetic_v<T>, Var &>
Var::operator+=(const T &ConstValue) {
  auto &Ctx = Fn.getFunction()->getContext();
  Var &Tmp = Fn.declVarInternal("tmp.", TypeMap<T>::get(Ctx));
  Tmp = ConstValue;

  *this += Tmp;

  return *this;
}

Var &Var::operator-=(Var &Other) {
  Var &Res = *this - Other;
  auto &IRB = Fn.getIRBuilder();
  this->storeValue(convert(IRB, Res.getValue(), getValueType()));

  return *this;
}

template <typename T>
std::enable_if_t<std::is_arithmetic_v<T>, Var &>
Var::operator-=(const T &ConstValue) {
  auto &Ctx = Fn.getFunction()->getContext();
  Var &Tmp = Fn.declVarInternal("tmp.", TypeMap<T>::get(Ctx));
  Tmp = ConstValue;

  *this -= Tmp;

  return *this;
}

Var &Var::operator*=(Var &Other) {
  Var &Res = *this * Other;
  auto &IRB = Fn.getIRBuilder();
  this->storeValue(convert(IRB, Res.getValue(), getValueType()));

  return *this;
}

template <typename T>
std::enable_if_t<std::is_arithmetic_v<T>, Var &>
Var::operator*=(const T &ConstValue) {
  auto &Ctx = Fn.getFunction()->getContext();
  Var &Tmp = Fn.declVarInternal("tmp.", TypeMap<T>::get(Ctx));
  Tmp = ConstValue;

  *this *= Tmp;

  return *this;
}

Var &Var::operator/=(Var &Other) {
  Var &Res = *this / Other;
  auto &IRB = Fn.getIRBuilder();
  this->storeValue(convert(IRB, Res.getValue(), getValueType()));

  return *this;
}

template <typename T>
std::enable_if_t<std::is_arithmetic_v<T>, Var &>
Var::operator/=(const T &ConstValue) {
  auto &Ctx = Fn.getFunction()->getContext();
  Var &Tmp = Fn.declVarInternal("tmp.", TypeMap<T>::get(Ctx));
  Tmp = ConstValue;

  *this /= Tmp;

  return *this;
}

Var &Var::operator%=(Var &Other) {
  Var &Res = *this % Other;
  auto &IRB = Fn.getIRBuilder();
  this->storeValue(convert(IRB, Res.getValue(), getValueType()));

  return *this;
}

template <typename T>
std::enable_if_t<std::is_arithmetic_v<T>, Var &>
Var::operator%=(const T &ConstValue) {
  auto &Ctx = Fn.getFunction()->getContext();
  Var &Tmp = Fn.declVarInternal("tmp.", TypeMap<T>::get(Ctx));
  Tmp = ConstValue;

  *this %= Tmp;

  return *this;
}

template <typename T>
std::enable_if_t<std::is_arithmetic_v<T>, Var &>
Var::operator+(const T &ConstValue) const {
  auto &Ctx = Fn.getFunction()->getContext();
  Var &Tmp = Fn.declVarInternal("tmp.", TypeMap<T>::get(Ctx));
  Tmp = ConstValue;
  return ((*this) + Tmp);
}

template <typename T>
std::enable_if_t<std::is_arithmetic_v<T>, Var &>
Var::operator-(const T &ConstValue) const {
  auto &Ctx = Fn.getFunction()->getContext();
  Var &Tmp = Fn.declVarInternal("tmp.", TypeMap<T>::get(Ctx));
  Tmp = ConstValue;
  return ((*this) - Tmp);
}

template <typename T>
std::enable_if_t<std::is_arithmetic_v<T>, Var &>
Var::operator*(const T &ConstValue) const {
  auto &Ctx = Fn.getFunction()->getContext();
  Var &Tmp = Fn.declVarInternal("tmp.", TypeMap<T>::get(Ctx));
  Tmp = ConstValue;
  return ((*this) * Tmp);
}

template <typename T>
std::enable_if_t<std::is_arithmetic_v<T>, Var &>
Var::operator/(const T &ConstValue) const {
  auto &Ctx = Fn.getFunction()->getContext();
  Var &Tmp = Fn.declVarInternal("tmp.", TypeMap<T>::get(Ctx));
  Tmp = ConstValue;
  return ((*this) / Tmp);
}

template <typename T>
std::enable_if_t<std::is_arithmetic_v<T>, Var &>
Var::operator%(const T &ConstValue) const {
  auto &Ctx = Fn.getFunction()->getContext();
  Var &Tmp = Fn.declVarInternal("tmp.", TypeMap<T>::get(Ctx));
  Tmp = ConstValue;
  return ((*this) % Tmp);
}

Var &Var::operator=(const Var &Other) {
  auto &IRB = Fn.getIRBuilder();
  Type *LHSType = getValueType();

  Value *RHS = convert(IRB, Other.getValue(), LHSType);
  storeValue(RHS);

  return *this;
}

template <typename T, typename> Var &Var::operator=(const T &ConstValue) {
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

/// Define comparison operators.

Var &Var::operator>(const Var &Other) const {
  return cmpOp(
      *this, Other,
      [](IRBuilderBase &B, Value *L, Value *R) {
        // TODO: fix for unsigned.
        return B.CreateICmpSGT(L, R);
      },
      [](IRBuilderBase &B, Value *L, Value *R) {
        return B.CreateFCmpOGT(L, R);
      });
}

Var &Var::operator>=(const Var &Other) const {
  return cmpOp(
      *this, Other,
      [](IRBuilderBase &B, Value *L, Value *R) {
        // TODO: fix for unsigned.
        return B.CreateICmpSGE(L, R);
      },
      [](IRBuilderBase &B, Value *L, Value *R) {
        return B.CreateFCmpOGE(L, R);
      });
}

Var &Var::operator<(const Var &Other) const {
  return cmpOp(
      *this, Other,
      [](IRBuilderBase &B, Value *L, Value *R) {
        // TODO: fix for unsigned.
        return B.CreateICmpSLT(L, R);
      },
      [](IRBuilderBase &B, Value *L, Value *R) {
        return B.CreateFCmpOLT(L, R);
      });
}

Var &Var::operator<=(const Var &Other) const {
  return cmpOp(
      *this, Other,
      [](IRBuilderBase &B, Value *L, Value *R) {
        // TODO: fix for unsigned.
        return B.CreateICmpSLE(L, R);
      },
      [](IRBuilderBase &B, Value *L, Value *R) {
        return B.CreateFCmpOLE(L, R);
      });
}

Var &Var::operator==(const Var &Other) const {
  return cmpOp(
      *this, Other,
      [](IRBuilderBase &B, Value *L, Value *R) {
        // TODO: fix for unsigned.
        return B.CreateICmpEQ(L, R);
      },
      [](IRBuilderBase &B, Value *L, Value *R) {
        return B.CreateFCmpOEQ(L, R);
      });
}

Var &Var::operator!=(const Var &Other) const {
  return cmpOp(
      *this, Other,
      [](IRBuilderBase &B, Value *L, Value *R) {
        // TODO: fix for unsigned.
        return B.CreateICmpNE(L, R);
      },
      [](IRBuilderBase &B, Value *L, Value *R) {
        return B.CreateFCmpONE(L, R);
      });
}

template <typename T>
std::enable_if_t<std::is_arithmetic_v<T>, Var &>
Var::operator>(const T &ConstValue) const {
  auto &Ctx = Fn.getFunction()->getContext();
  Var &Tmp = Fn.declVarInternal("cmp.", TypeMap<T>::get(Ctx));
  Tmp = ConstValue;

  return ((*this) > Tmp);
}

template <typename T>
std::enable_if_t<std::is_arithmetic_v<T>, Var &>
Var::operator>=(const T &ConstValue) const {
  auto &Ctx = Fn.getFunction()->getContext();
  Var &Tmp = Fn.declVarInternal("cmp.", TypeMap<T>::get(Ctx));
  Tmp = ConstValue;

  return ((*this) >= Tmp);
}

template <typename T>
std::enable_if_t<std::is_arithmetic_v<T>, Var &>
Var::operator<(const T &ConstValue) const {
  auto &Ctx = Fn.getFunction()->getContext();
  Var &Tmp = Fn.declVarInternal("cmp.", TypeMap<T>::get(Ctx));
  Tmp = ConstValue;

  return ((*this) < Tmp);
}

template <typename T>
std::enable_if_t<std::is_arithmetic_v<T>, Var &>
Var::operator<=(const T &ConstValue) const {
  auto &Ctx = Fn.getFunction()->getContext();
  Var &Tmp = Fn.declVarInternal("cmp.", TypeMap<T>::get(Ctx));
  Tmp = ConstValue;

  return ((*this) <= Tmp);
}

template <typename T>
std::enable_if_t<std::is_arithmetic_v<T>, Var &>
Var::operator==(const T &ConstValue) const {
  auto &Ctx = Fn.getFunction()->getContext();
  Var &Tmp = Fn.declVarInternal("cmp.", TypeMap<T>::get(Ctx));
  Tmp = ConstValue;

  return ((*this) == Tmp);
}

template <typename T>
std::enable_if_t<std::is_arithmetic_v<T>, Var &>
Var::operator!=(const T &ConstValue) const {
  auto &Ctx = Fn.getFunction()->getContext();
  Var &Tmp = Fn.declVarInternal("cmp.", TypeMap<T>::get(Ctx));
  Tmp = ConstValue;

  return ((*this) != Tmp);
}

/// End of comparison operators.

VarKind Var::kind() const { return Kind; }

Var &Var::operator[](size_t I) { return index(I); }

Var &Var::operator[](const Var &IdxVar) { return index(IdxVar); }

// Define non-member operators.

template <typename T>
std::enable_if_t<std::is_arithmetic_v<T>, Var &> operator+(const T &ConstValue,
                                                           const Var &V) {
  auto &Ctx = V.Fn.getFunction()->getContext();
  Var &Tmp = V.Fn.declVarInternal("tmp.", TypeMap<T>::get(Ctx));
  Tmp = ConstValue;

  return (Tmp + V);
}

template <typename T>
std::enable_if_t<std::is_arithmetic_v<T>, Var &> operator-(const T &ConstValue,
                                                           const Var &V) {
  auto &Ctx = V.Fn.getFunction()->getContext();
  Var &Tmp = V.Fn.declVarInternal("tmp.", TypeMap<T>::get(Ctx));
  Tmp = ConstValue;

  return (Tmp - V);
}

template <typename T>
std::enable_if_t<std::is_arithmetic_v<T>, Var &> operator*(const T &ConstValue,
                                                           const Var &V) {
  auto &Ctx = V.Fn.getFunction()->getContext();
  Var &Tmp = V.Fn.declVarInternal("tmp.", TypeMap<T>::get(Ctx));
  Tmp = ConstValue;

  return (Tmp * V);
}

template <typename T>
std::enable_if_t<std::is_arithmetic_v<T>, Var &> operator/(const T &ConstValue,
                                                           const Var &V) {
  auto &Ctx = V.Fn.getFunction()->getContext();
  Var &Tmp = V.Fn.declVarInternal("tmp.", TypeMap<T>::get(Ctx));
  Tmp = ConstValue;

  return (Tmp / V);
}

template <typename T>
std::enable_if_t<std::is_arithmetic_v<T>, Var &> operator%(const T &ConstValue,
                                                           const Var &V) {
  auto &Ctx = V.Fn.getFunction()->getContext();
  Var &Tmp = V.Fn.declVarInternal("tmp.", TypeMap<T>::get(Ctx));
  Tmp = ConstValue;

  return (Tmp % V);
}

Value *convert(IRBuilderBase IRB, Value *V, Type *TargetType) {
  Type *ValType = V->getType();

  if (ValType == TargetType) {
    return V;
  }

  if (ValType->isIntegerTy() && TargetType->isFloatingPointTy()) {
    return IRB.CreateSIToFP(V, TargetType);
  }

  if (ValType->isFloatingPointTy() && TargetType->isIntegerTy()) {
    return IRB.CreateFPToSI(V, TargetType);
  }

  if (ValType->isIntegerTy() && TargetType->isIntegerTy()) {
    if (ValType->getIntegerBitWidth() < TargetType->getIntegerBitWidth())
      // TODO: emit the correct signed variant.
      return IRB.CreateIntCast(V, TargetType, false);

    // Truncate if Val has more bits than the target type.
    return IRB.CreateTrunc(V, TargetType);
  }

  if (ValType->isFloatingPointTy() && TargetType->isFloatingPointTy())
    return IRB.CreateFPExt(V, TargetType);

  PROTEUS_FATAL_ERROR("Unsupported conversion");
}

/// Get the common type following C++ usual arithmetic conversions.
Type *getCommonType(const DataLayout &DL, Type *T1, Type *T2) {
  // Give priority to floating point types.
  if (T1->isFloatingPointTy() && T2->isIntegerTy()) {
    return T1;
  }

  if (T2->isFloatingPointTy() && T1->isIntegerTy())
    return T2;

  // Return the wider integer type.
  if (T1->isIntegerTy() && T2->isIntegerTy()) {
    return ((T1->getIntegerBitWidth() >= T2->getIntegerBitWidth()) ? T1 : T2);
  }

  if (T1->isFloatingPointTy() && T2->isFloatingPointTy()) {
    return ((DL.getTypeSizeInBits(T1) >= DL.getTypeSizeInBits(T2)) ? T1 : T2);
  }

  PROTEUS_FATAL_ERROR("Unsupported conversion types");
}

Var &powf(const Var &L, const Var &R) {
  auto &Fn = L.Fn;
  auto &M = *Fn.getFunction()->getParent();
  auto &IRB = Fn.getIRBuilder();

  auto *ResultType = IRB.getFloatTy();
  Var &ResultVar = Fn.declVarInternal("res.", ResultType);

#if PROTEUS_ENABLE_CUDA
  std::string IntrinsicName = "__nv_powf";
#else
  std::string IntrinsicName = "llvm.pow.f32";
#endif

  FunctionCallee Callee =
      M.getOrInsertFunction(IntrinsicName, ResultType, ResultType, ResultType);
  auto *Call = IRB.CreateCall(Callee, {convert(IRB, L.getValue(), ResultType),
                                       convert(IRB, R.getValue(), ResultType)});
  ResultVar.storeValue(Call);

  return ResultVar;
}

Var &sqrtf(const Var &R) {
  auto &Fn = R.Fn;
  auto &M = *Fn.getFunction()->getParent();
  auto &IRB = Fn.getIRBuilder();

  auto *ResultType = IRB.getFloatTy();
  Var &ResultVar = Fn.declVarInternal("res.", ResultType);

#if PROTEUS_ENABLE_CUDA
  std::string IntrinsicName = "__nv_sqrtf";
#else
  std::string IntrinsicName = "llvm.sqrt.f32";
#endif

  FunctionCallee Callee =
      M.getOrInsertFunction(IntrinsicName, ResultType, ResultType);
  auto *Call = IRB.CreateCall(Callee, {R.getValue()});
  ResultVar.storeValue(Call);

  return ResultVar;
}

Var &min(const Var &L, const Var &R) {
  FuncBase &Fn = L.Fn;
  if (&Fn != &R.Fn)
    PROTEUS_FATAL_ERROR("Variables should belong to the same function");

  Var &ResultVar = Fn.declVarInternal("res.", L.getValueType());
  ResultVar = R;
  Fn.beginIf(L < R);
  { ResultVar = L; }
  Fn.endIf();
  return ResultVar;
}

// Assignment explicit instantiations.
template Var &Var::operator= <int>(const int &);
template Var &Var::operator= <unsigned int>(const unsigned int &);
template Var &Var::operator= <size_t>(const size_t &);
template Var &Var::operator= <float>(const float &);
template Var &Var::operator= <double>(const double &);

// Binary operators explicit instantiations.
template Var &Var::operator+ <int>(const int &) const;
template Var &Var::operator+ <unsigned int>(const unsigned int &) const;
template Var &Var::operator+ <size_t>(const size_t &) const;
template Var &Var::operator+ <float>(const float &) const;
template Var &Var::operator+ <double>(const double &) const;

template Var &Var::operator- <int>(const int &) const;
template Var &Var::operator- <unsigned int>(const unsigned int &) const;
template Var &Var::operator- <size_t>(const size_t &) const;
template Var &Var::operator- <float>(const float &) const;
template Var &Var::operator- <double>(const double &) const;

template Var &Var::operator* <int>(const int &) const;
template Var &Var::operator* <unsigned int>(const unsigned int &) const;
template Var &Var::operator* <size_t>(const size_t &) const;
template Var &Var::operator* <float>(const float &) const;
template Var &Var::operator* <double>(const double &) const;

template Var &Var::operator/ <int>(const int &) const;
template Var &Var::operator/ <unsigned int>(const unsigned int &) const;
template Var &Var::operator/ <size_t>(const size_t &) const;
template Var &Var::operator/ <float>(const float &) const;
template Var &Var::operator/ <double>(const double &) const;

template Var &Var::operator% <int>(const int &) const;
template Var &Var::operator% <unsigned int>(const unsigned int &) const;
template Var &Var::operator% <size_t>(const size_t &) const;
template Var &Var::operator% <float>(const float &) const;
template Var &Var::operator% <double>(const double &) const;

// Binary operators with assignment explicit instantiations.
template Var &Var::operator+= <int>(const int &);
template Var &Var::operator+= <unsigned int>(const unsigned int &);
template Var &Var::operator+= <size_t>(const size_t &);
template Var &Var::operator+= <float>(const float &);
template Var &Var::operator+= <double>(const double &);

template Var &Var::operator-= <int>(const int &);
template Var &Var::operator-= <unsigned int>(const unsigned int &);
template Var &Var::operator-= <size_t>(const size_t &);
template Var &Var::operator-= <float>(const float &);
template Var &Var::operator-= <double>(const double &);

template Var &Var::operator*= <int>(const int &);
template Var &Var::operator*= <unsigned int>(const unsigned int &);
template Var &Var::operator*= <size_t>(const size_t &);
template Var &Var::operator*= <float>(const float &);
template Var &Var::operator*= <double>(const double &);

template Var &Var::operator/= <int>(const int &);
template Var &Var::operator/= <unsigned int>(const unsigned int &);
template Var &Var::operator/= <size_t>(const size_t &);
template Var &Var::operator/= <float>(const float &);
template Var &Var::operator/= <double>(const double &);

template Var &Var::operator%= <int>(const int &);
template Var &Var::operator%= <unsigned int>(const unsigned int &);
template Var &Var::operator%= <size_t>(const size_t &);
template Var &Var::operator%= <float>(const float &);
template Var &Var::operator%= <double>(const double &);

// Non-member binary operator explicit instantiations.
template Var &operator+ <int>(const int &, const Var &);
template Var &operator+ <unsigned int>(const unsigned int &, const Var &);
template Var &operator+ <size_t>(const size_t &, const Var &);
template Var &operator+ <float>(const float &, const Var &);
template Var &operator+ <double>(const double &, const Var &);

template Var &operator- <int>(const int &, const Var &);
template Var &operator- <unsigned int>(const unsigned int &, const Var &);
template Var &operator- <size_t>(const size_t &, const Var &);
template Var &operator- <float>(const float &, const Var &);
template Var &operator- <double>(const double &, const Var &);

template Var &operator* <int>(const int &, const Var &);
template Var &operator* <unsigned int>(const unsigned int &, const Var &);
template Var &operator* <size_t>(const size_t &, const Var &);
template Var &operator* <float>(const float &, const Var &);
template Var &operator* <double>(const double &, const Var &);

template Var &operator/ <int>(const int &, const Var &);
template Var &operator/ <unsigned int>(const unsigned int &, const Var &);
template Var &operator/ <size_t>(const size_t &, const Var &);
template Var &operator/ <float>(const float &, const Var &);
template Var &operator/ <double>(const double &, const Var &);

template Var &operator% <int>(const int &, const Var &);
template Var &operator% <unsigned int>(const unsigned int &, const Var &);
template Var &operator% <size_t>(const size_t &, const Var &);
template Var &operator% <float>(const float &, const Var &);
template Var &operator% <double>(const double &, const Var &);

// Comparison explicit instantiations.
template Var &Var::operator>(const int &ConstValue) const;
template Var &Var::operator>(const unsigned int &ConstValue) const;
template Var &Var::operator>(const float &ConstValue) const;
template Var &Var::operator>(const double &ConstValue) const;

template Var &Var::operator>=(const int &ConstValue) const;
template Var &Var::operator>=(const unsigned int &ConstValue) const;
template Var &Var::operator>=(const float &ConstValue) const;
template Var &Var::operator>=(const double &ConstValue) const;

template Var &Var::operator<(const int &ConstValue) const;
template Var &Var::operator<(const unsigned int &ConstValue) const;
template Var &Var::operator<(const float &ConstValue) const;
template Var &Var::operator<(const double &ConstValue) const;

template Var &Var::operator<=(const int &ConstValue) const;
template Var &Var::operator<=(const unsigned int &ConstValue) const;
template Var &Var::operator<=(const float &ConstValue) const;
template Var &Var::operator<=(const double &ConstValue) const;

template Var &Var::operator==(const int &ConstValue) const;
template Var &Var::operator==(const unsigned int &ConstValue) const;
template Var &Var::operator==(const float &ConstValue) const;
template Var &Var::operator==(const double &ConstValue) const;

ScalarVar::ScalarVar(AllocaInst *Slot, FuncBase &Fn)
    : Var(Slot, Fn), Slot(Slot) {
  Kind = VarKind::Scalar;
}

StringRef ScalarVar::getName() const { return Slot->getName(); }

Type *ScalarVar::getValueType() const { return Slot->getAllocatedType(); }

Value *ScalarVar::getValue() const {
  auto &IRB = Fn.getIRBuilder();
  return IRB.CreateLoad(Slot->getAllocatedType(), Slot);
}

void ScalarVar::storeValue(Value *Val) {
  auto &IRB = Fn.getIRBuilder();
  IRB.CreateStore(Val, Slot);
}

VarKind ScalarVar::kind() const { return VarKind::Scalar; }

AllocaInst *ScalarVar::getAlloca() const { return Slot; }

Value *ScalarVar::getPointerValue() const {
  PROTEUS_FATAL_ERROR("ScalarVar does not hold a pointer");
}

void ScalarVar::storePointer(Value * /*Ptr*/) {
  PROTEUS_FATAL_ERROR("ScalarVar does not hold a pointer");
}

Var &ScalarVar::index(size_t /*I*/) {
  PROTEUS_FATAL_ERROR("ScalarVar does not support indexing");
}

Var &ScalarVar::index(const Var & /*I*/) {
  PROTEUS_FATAL_ERROR("ScalarVar does not support indexing");
}

Var &ScalarVar::getAddress() {
  Type *ScalarTy = Slot->getAllocatedType();
  auto *BasePtrTy = cast<PointerType>(Slot->getType());
  unsigned AddrSpace = BasePtrTy->getAddressSpace();
  Type *PtrTy = PointerType::get(ScalarTy, AddrSpace);

  auto &ResultVar = Fn.declVarInternal("addr.", PtrTy, ScalarTy);
  ResultVar.storePointer(Slot);
  return ResultVar;
}

PointerVar::PointerVar(AllocaInst *PtrSlot, FuncBase &Fn, Type *ElemTy)
    : Var(PtrSlot, Fn), PointerElemTy(ElemTy) {
  Kind = VarKind::Pointer;
}

StringRef PointerVar::getName() const { return Alloca->getName(); }

Type *PointerVar::getValueType() const { return PointerElemTy; }

Value *PointerVar::getPointerValue() const {
  auto &IRB = Fn.getIRBuilder();
  Type *SlotTy = Alloca->getAllocatedType();
  if (!SlotTy->isPointerTy())
    PROTEUS_FATAL_ERROR("PointerVar alloca must allocate a pointer type");
  return IRB.CreateLoad(SlotTy, Alloca);
}

void PointerVar::storePointer(Value *Ptr) {
  auto &IRB = Fn.getIRBuilder();
  IRB.CreateStore(Ptr, Alloca);
}

Value *PointerVar::getValue() const {
  auto &IRB = Fn.getIRBuilder();
  Value *Ptr = getPointerValue();
  return IRB.CreateLoad(PointerElemTy, Ptr);
}

void PointerVar::storeValue(Value *Val) {
  auto &IRB = Fn.getIRBuilder();
  // TODO: This is too permissive and allows assigning a value to a pointer's
  // memory location, e.g:
  // ...
  // Var &V = declVar<double *>();
  // V = 42 <--- Will store 42 to the memory location pointed by V!
  // ...
  // Fix for compliance with C++ typing and rules, use traits and compile-time
  // processing when possible.
  Value *Ptr = getPointerValue();
  IRB.CreateStore(Val, Ptr);
}

VarKind PointerVar::kind() const { return VarKind::Pointer; }

AllocaInst *PointerVar::getAlloca() const { return Alloca; }

Var &PointerVar::index(size_t I) {
  auto &IRB = Fn.getIRBuilder();

  auto *Ptr = IRB.CreateLoad(Alloca->getAllocatedType(), Alloca);
  auto *GEP = IRB.CreateConstInBoundsGEP1_64(PointerElemTy, Ptr, I);
  auto *BasePtrTy = cast<PointerType>(Ptr->getType());
  unsigned AddrSpace = BasePtrTy->getAddressSpace();
  Type *ElemPtrTy = PointerType::get(PointerElemTy, AddrSpace);

  auto &ResultVar = Fn.declVarInternal("res.", ElemPtrTy, PointerElemTy);
  ResultVar.storePointer(GEP);
  return ResultVar;
}

Var &PointerVar::index(const Var &I) {
  auto &IRB = Fn.getIRBuilder();

  auto *Ptr = IRB.CreateLoad(Alloca->getAllocatedType(), Alloca);
  auto *GEP = IRB.CreateInBoundsGEP(PointerElemTy, Ptr, I.getValue());
  auto *BasePtrTy = cast<PointerType>(Ptr->getType());
  unsigned AddrSpace = BasePtrTy->getAddressSpace();
  Type *ElemPtrTy = PointerType::get(PointerElemTy, AddrSpace);

  auto &ResultVar = Fn.declVarInternal("res.", ElemPtrTy, PointerElemTy);
  ResultVar.storePointer(GEP);
  return ResultVar;
}

Var &PointerVar::getAddress() {
  // Alloca->getType() is T*.
  auto *BasePtrTy = cast<PointerType>(Alloca->getType());
  unsigned AddrSpace = BasePtrTy->getAddressSpace();
  // ElemPtrTy represents the type T**.
  Type *ElemPtrTy = PointerType::get(BasePtrTy, AddrSpace);
  auto &ResultVar = Fn.declVarInternal("addr.", ElemPtrTy, BasePtrTy);
  ResultVar.storePointer(Alloca);
  return ResultVar;

}

ArrayVar::ArrayVar(Value *BasePointer, FuncBase &Fn, ArrayType *ArrayTy)
    : Var(Fn), BasePointer(BasePointer), ArrayTy(ArrayTy) {
  Kind = VarKind::Array;
}

StringRef ArrayVar::getName() const { return BasePointer->getName(); }

Type *ArrayVar::getValueType() const { return ArrayTy; }

Value *ArrayVar::getValue() const {
  PROTEUS_FATAL_ERROR(
      "ArrayVar does not support load/store of aggregate value");
}

void ArrayVar::storeValue(Value *Val) {
  (void)Val;
  PROTEUS_FATAL_ERROR(
      "ArrayVar does not support load/store of aggregate value");
}

Value *ArrayVar::getPointerValue() const {
  PROTEUS_FATAL_ERROR("ArrayVar does not support getPointerValue");
}

void ArrayVar::storePointer(Value * /*Ptr*/) {
  PROTEUS_FATAL_ERROR("ArrayVar does not support storePointer");
}

VarKind ArrayVar::kind() const { return VarKind::Array; }

Var &ArrayVar::index(size_t I) {
  auto &IRB = Fn.getIRBuilder();
  // GEP into the array aggregate: [0, I]
  auto *GEP = IRB.CreateConstInBoundsGEP2_64(ArrayTy, BasePointer, 0, I);
  Type *ElemTy = ArrayTy->getArrayElementType();
  auto *BasePtrTy = cast<PointerType>(BasePointer->getType());
  unsigned AddrSpace = BasePtrTy->getAddressSpace();
  Type *ElemPtrTy = PointerType::get(ElemTy, AddrSpace);

  auto &ResultVar = Fn.declVarInternal("res.", ElemPtrTy, ElemTy);
  ResultVar.storePointer(GEP);
  return ResultVar;
}

Var &ArrayVar::index(const Var &I) {
  auto &IRB = Fn.getIRBuilder();
  Value *IdxVal = I.getValue();
  if (!IdxVal->getType()->isIntegerTy())
    PROTEUS_FATAL_ERROR("Expected integer index for array GEP");
  Value *Zero = llvm::ConstantInt::get(IdxVal->getType(), 0);
  auto *GEP = IRB.CreateInBoundsGEP(ArrayTy, BasePointer, {Zero, IdxVal});
  Type *ElemTy = ArrayTy->getArrayElementType();
  auto *BasePtrTy = cast<PointerType>(BasePointer->getType());
  unsigned AddrSpace = BasePtrTy->getAddressSpace();
  Type *ElemPtrTy = PointerType::get(ElemTy, AddrSpace);

  auto &ResultVar = Fn.declVarInternal("res.", ElemPtrTy, ElemTy);
  ResultVar.storePointer(GEP);
  return ResultVar;
}

Var &ArrayVar::getAddress() {
  PROTEUS_FATAL_ERROR("ArrayVar does not support getAddress");
}

} // namespace proteus
