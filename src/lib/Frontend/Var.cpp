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

Value *AllocaStorage::getValue(IRBuilderBase &IRB) const {
  Type *AllocaType = Alloca->getAllocatedType();
  if (AllocaType->isPointerTy()) {
    auto *Ptr = IRB.CreateLoad(AllocaType, Alloca);
    return IRB.CreateLoad(PointerElemType, Ptr);
  }
  return IRB.CreateLoad(AllocaType, Alloca);
}

Type *AllocaStorage::getValueType() const {
  Type *AllocaType = Alloca->getAllocatedType();
  if (AllocaType->isPointerTy()) {
    return PointerElemType;
  }
  return AllocaType;
}

StringRef AllocaStorage::getName() const { return Alloca->getName(); }

void AllocaStorage::storeValue(IRBuilderBase &IRB, Value *Val) {
  Type *AllocaType = Alloca->getAllocatedType();
  if (AllocaType->isPointerTy()) {
    auto *Ptr = IRB.CreateLoad(AllocaType, Alloca);
    IRB.CreateStore(Val, Ptr);
  } else {
    IRB.CreateStore(Val, Alloca);
  }
}

void AllocaStorage::storePointer(IRBuilderBase &IRB, Value *Ptr) {
  if (!isPointer())
    PROTEUS_FATAL_ERROR("Expected pointer type");
  IRB.CreateStore(Ptr, Alloca);
}

bool AllocaStorage::isPointer() const {
  Type *AllocaType = Alloca->getAllocatedType();
  if (AllocaType->isPointerTy()) {
    if (!PointerElemType)
      PROTEUS_FATAL_ERROR("Expected pointer type");
    return true;
  }
  return false;
}

Value *BorrowedStorage::getValue(IRBuilderBase &IRB) const {
  return IRB.CreateLoad(PointerElemType, PointerValue);
}

Type *BorrowedStorage::getValueType() const { return PointerElemType; }

StringRef BorrowedStorage::getName() const { return PointerValue->getName(); }

void BorrowedStorage::storeValue(IRBuilderBase &IRB, Value *Val) {
  IRB.CreateStore(Val, PointerValue);
}

void BorrowedStorage::storePointer(IRBuilderBase &IRB, Value *Ptr) {
  IRB.CreateStore(Ptr, PointerValue);
}

bool BorrowedStorage::isPointer() const { return true; }

Var::Var(AllocaInst *Alloca, FuncBase &Fn, Type *PointerElemType)
    : Storage(AllocaStorage{Alloca, PointerElemType}), Fn(Fn) {}

Var::Var(Value *PointerValue, FuncBase &Fn, Type *PointerElemType)
    : Storage(BorrowedStorage{PointerValue, PointerElemType}), Fn(Fn) {}

Var Var::fromBorrowed(Value *PointerValue, FuncBase &Fn,
                      Type *PointerElemType) {
  return Var(PointerValue, Fn, PointerElemType);
}

Value *Var::getValue() const {
  auto &IRB = Fn.getIRBuilder();
  return std::visit([&](const auto &Storage) { return Storage.getValue(IRB); },
                    Storage);
}

Type *Var::getValueType() const {
  return std::visit([](const auto &Storage) { return Storage.getValueType(); },
                    Storage);
}

StringRef Var::getName() {
  return std::visit([](const auto &Storage) { return Storage.getName(); },
                    Storage);
}

bool Var::isPointer() const {
  return std::visit([](const auto &Storage) { return Storage.isPointer(); },
                    Storage);
}

void Var::storeValue(Value *Val) {
  auto &IRB = Fn.getIRBuilder();
  std::visit([&](auto &Storage) { Storage.storeValue(IRB, Val); }, Storage);
}

void Var::storePointer(Value *Ptr) {
  auto &IRB = Fn.getIRBuilder();
  std::visit([&](auto &Storage) { Storage.storePointer(IRB, Ptr); }, Storage);
}

AllocaInst *Var::getAlloca() const {
  return std::visit(
      [](const auto &Storage) -> AllocaInst * {
        if constexpr (std::is_same_v<std::decay_t<decltype(Storage)>,
                                     AllocaStorage>) {
          return Storage.Alloca;
        } else {
          PROTEUS_FATAL_ERROR("Expected AllocaStorage for getAlloca()");
        }
      },
      Storage);
}

Type *Var::getPointerElemType() const {
  return std::visit([](const auto &Storage) { return Storage.PointerElemType; },
                    Storage);
}

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

  *this += Tmp;

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

  return ((*this) < Tmp);
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

Var &Var::operator[](size_t I) {
  auto &IRB = Fn.getIRBuilder();

  if (!isPointer())
    PROTEUS_FATAL_ERROR("Expected pointer type: Var " + getName());

  return std::visit(
      [&](const auto &Storage) -> Var & {
        Type *PointerElemType = Storage.PointerElemType;
        auto &ResultVar = Fn.declVarInternal(
            "res.", PointerElemType->getPointerTo(), PointerElemType);

        if constexpr (std::is_same_v<std::decay_t<decltype(Storage)>,
                                     AllocaStorage>) {
          auto *Ptr = IRB.CreateLoad(Storage.Alloca->getAllocatedType(),
                                     Storage.Alloca);
          auto *GEP = IRB.CreateConstInBoundsGEP1_64(PointerElemType, Ptr, I);
          ResultVar.storePointer(GEP);
        } else {
          // BorrowedStorage always points to scalar array element.
          PROTEUS_FATAL_ERROR("Expected AllocaStorage for operator[]");
        }

        return ResultVar;
      },
      Storage);
}

Var &Var::operator[](const Var &IdxVar) {
  auto &IRB = Fn.getIRBuilder();

  if (!isPointer())
    PROTEUS_FATAL_ERROR("Expected pointer type");

  return std::visit(
      [&](const auto &Storage) -> Var & {
        Type *PointerElemType = Storage.PointerElemType;
        auto &ResultVar = Fn.declVarInternal(
            "res.", PointerElemType->getPointerTo(), PointerElemType);
        Value *Idx = IdxVar.getValue();

        if constexpr (std::is_same_v<std::decay_t<decltype(Storage)>,
                                     AllocaStorage>) {
          auto *Ptr = IRB.CreateLoad(Storage.Alloca->getAllocatedType(),
                                     Storage.Alloca);
          auto *GEP = IRB.CreateInBoundsGEP(PointerElemType, Ptr, {Idx});
          ResultVar.storePointer(GEP);
        } else {
          auto *GEP = IRB.CreateInBoundsGEP(PointerElemType,
                                            Storage.PointerValue, {Idx});
          ResultVar.storePointer(GEP);
        }

        return ResultVar;
      },
      Storage);
}

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
  {
    ResultVar = L;
  }
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

} // namespace proteus
