#ifndef PROTEUS_FRONTEND_FUNC_HPP
#define PROTEUS_FRONTEND_FUNC_HPP

#include <memory>

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

#include "proteus/AddressSpace.hpp"
#include "proteus/Error.h"
#include "proteus/Frontend/Dispatcher.hpp"
#include "proteus/Frontend/TypeMap.hpp"
#include "proteus/Frontend/Var.hpp"
#include "proteus/Frontend/VarStorage.hpp"

namespace proteus {

class JitModule;
template <typename T> class LoopBoundInfo;
template <typename T, typename... ForLoopBuilders> class LoopNestBuilder;
template <typename T, typename BodyLambda> class ForLoopBuilder;

// Helper struct to represent the signature of a function.
// Useful to partially-specialize function templates.
template <typename... ArgTs> struct ArgTypeList {};
template <typename T> struct FnSig;
template <typename RetT_, typename... ArgT> struct FnSig<RetT_(ArgT...)> {
  using ArgsTList = ArgTypeList<ArgT...>;
  using RetT = RetT_;
};

using namespace llvm;

struct EmptyLambda {
  void operator()() const {}
};

class FuncBase {
protected:
  JitModule &J;
  FunctionCallee FC;
  IRBuilder<> IRB;
  IRBuilderBase::InsertPoint IP;

  std::string Name;

  enum class ScopeKind { FUNCTION, IF, FOR };
  struct Scope {
    std::string File;
    int Line;
    ScopeKind Kind;
    IRBuilderBase::InsertPoint ContIP;

    explicit Scope(const char *File, int Line, ScopeKind Kind,
                   IRBuilderBase::InsertPoint ContIP)
        : File(File), Line(Line), Kind(Kind), ContIP(ContIP) {}
  };
  std::vector<Scope> Scopes;

  std::string toString(ScopeKind Kind) {
    switch (Kind) {
    case ScopeKind::FUNCTION:
      return "FUNCTION";
    case ScopeKind::IF:
      return "IF";
    case ScopeKind::FOR:
      return "FOR";
    default:
      PROTEUS_FATAL_ERROR("Unsupported Kind " +
                          std::to_string(static_cast<int>(Kind)));
    }
  }

  template <typename T>
  Var<T> emitAtomic(AtomicRMWInst::BinOp Op, const Var<T *> &Addr,
                    const Var<T> &Val);

public:
  FuncBase(JitModule &J, FunctionCallee FC);

  Function *getFunction();

  AllocaInst *emitAlloca(Type *Ty, StringRef Name,
                         AddressSpace AS = AddressSpace::DEFAULT);

  Value *emitArrayCreate(Type *Ty, AddressSpace AT, StringRef Name);

  IRBuilderBase &getIRBuilder();

  template <typename T> Var<T> declVarInternal(StringRef Name = "var") {
    static_assert(!std::is_array_v<T>, "Expected non-array type");

    Function *F = getFunction();
    auto &Ctx = F->getContext();
    Type *AllocaTy = TypeMap<T>::get(Ctx);
    auto *Alloca = emitAlloca(AllocaTy, Name);

    if constexpr (std::is_pointer_v<T>) {
      Type *PtrElemTy = TypeMap<T>::getPointerElemType(Ctx);
      return Var<T>(std::make_unique<PointerStorage>(Alloca, IRB, PtrElemTy),
                    *this);
    } else {
      return Var<T>(std::make_unique<ScalarStorage>(Alloca, IRB), *this);
    }
  }

  template <typename T> Var<T> declVar(StringRef Name = "var") {
    static_assert(!std::is_array_v<T>, "Expected non-array type");
    return declVarInternal<T>(Name);
  }

  template <typename T>
  Var<T> declVar(size_t NElem, AddressSpace AS = AddressSpace::DEFAULT,
                 StringRef Name = "array_var") {
    static_assert(std::is_array_v<T>, "Expected array type");

    Function *F = getFunction();
    auto *BasePointer =
        emitArrayCreate(TypeMap<T>::get(F->getContext(), NElem), AS, Name);

    auto *ArrTy = cast<ArrayType>(TypeMap<T>::get(F->getContext(), NElem));
    return Var<T>(std::make_unique<ArrayStorage>(BasePointer, IRB, ArrTy),
                  *this);
  }

  template <typename T> Var<T> defVar(T Val, StringRef Name = "var") {
    Var<T> Var = declVarInternal<T>(Name);
    Var = Val;
    return Var;
  }

  template <typename T, typename U>
  Var<T> defVar(const Var<U> &Var, StringRef Name = "var") {
    auto Res = declVarInternal<T>(Name);
    Res = Var;
    return Res;
  }

  template <typename T>
  Var<T> defRuntimeConst(T Val, StringRef Name = "run.const.var") {
    return defVar<T>(Val, Name);
  }

  template <typename... ArgT> auto defRuntimeConsts(ArgT &&...Args) {
    return std::make_tuple(defRuntimeConst(std::forward<ArgT>(Args))...);
  }

  void beginFunction(const char *File = __builtin_FILE(),
                     int Line = __builtin_LINE());
  void endFunction();

  void beginIf(const Var<bool> &CondVar, const char *File = __builtin_FILE(),
               int Line = __builtin_LINE());
  void endIf();

  template <typename T>
  void beginFor(Var<T> &IterVar, const Var<T> &InitVar,
                const Var<T> &UpperBound, const Var<T> &IncVar,
                const char *File = __builtin_FILE(),
                int Line = __builtin_LINE());
  void endFor();

  template <typename Sig>
  std::enable_if_t<!std::is_void_v<typename FnSig<Sig>::RetT>,
                   Var<typename FnSig<Sig>::RetT>>
  call(StringRef Name);

  template <typename Sig>
  std::enable_if_t<std::is_void_v<typename FnSig<Sig>::RetT>, void>
  call(StringRef Name);

  template <typename Sig, typename... ArgVars>
  std::enable_if_t<!std::is_void_v<typename FnSig<Sig>::RetT>,
                   Var<typename FnSig<Sig>::RetT>>
  call(StringRef Name, ArgVars &&...ArgsVars);

  template <typename Sig, typename... ArgVars>
  std::enable_if_t<std::is_void_v<typename FnSig<Sig>::RetT>, void>
  call(StringRef Name, ArgVars &&...ArgsVars);

  template <typename BuiltinFuncT>
  decltype(auto) callBuiltin(BuiltinFuncT &&BuiltinFunc) {
    using RetT = std::invoke_result_t<BuiltinFuncT &, FuncBase &>;
    if constexpr (std::is_void_v<RetT>) {
      std::invoke(std::forward<BuiltinFuncT>(BuiltinFunc), *this);
    } else {
      return std::invoke(std::forward<BuiltinFuncT>(BuiltinFunc), *this);
    }
  }

  template <typename T>
  std::enable_if_t<std::is_arithmetic_v<T>, Var<T>>
  atomicAdd(const Var<T *> &Addr, const Var<T> &Val);
  template <typename T>
  std::enable_if_t<std::is_arithmetic_v<T>, Var<T>>
  atomicSub(const Var<T *> &Addr, const Var<T> &Val);
  template <typename T>
  std::enable_if_t<std::is_arithmetic_v<T>, Var<T>>
  atomicMax(const Var<T *> &Addr, const Var<T> &Val);
  template <typename T>
  std::enable_if_t<std::is_arithmetic_v<T>, Var<T>>
  atomicMin(const Var<T *> &Addr, const Var<T> &Val);

  template <typename T, typename BodyLambda = EmptyLambda>
  auto forLoop(const LoopBoundInfo<T> &Bounds, BodyLambda &&Body = {}) {
    return ForLoopBuilder<T, BodyLambda>(Bounds, *this, std::move(Body));
  }

  template <typename... LoopBuilders>
  auto buildLoopNest(LoopBuilders &&...Loops) {
    using FirstBuilder = std::remove_reference_t<
        std::tuple_element_t<0, std::tuple<LoopBuilders...>>>;
    using T = typename FirstBuilder::LoopIndexType;
    return LoopNestBuilder<T, std::decay_t<LoopBuilders>...>(
        *this, std::forward<LoopBuilders>(Loops)...);
  }

  template <typename T> void ret(const Var<T> &RetVal);

  void ret();

  StringRef getName() const { return Name; }

  void setName(StringRef NewName) {
    Name = NewName.str();
    Function *F = getFunction();
    F->setName(Name);
  }

  // Convert the given Var's value to type U and return a new Var holding
  // the converted value.
  template <typename U, typename T>
  std::enable_if_t<std::is_convertible_v<T, U>, Var<U>>
  convert(const Var<T> &V) {
    auto &IRBRef = getIRBuilder();
    Var<U> Res = declVarInternal<U>("convert.");
    Value *Converted = proteus::convert<T, U>(IRBRef, V.loadValue());
    Res.storeValue(Converted);
    return Res;
  }
};

template <typename RetT, typename... ArgT> class Func final : public FuncBase {
private:
  Dispatcher &Dispatch;
  RetT (*CompiledFunc)(ArgT...) = nullptr;
  // Optional because ArgTT is not default constructible.
  std::tuple<std::optional<Var<ArgT>>...> ArgumentsTT;

private:
  template <typename T, std::size_t ArgIdx> Var<T> createArg() {
    Function *F = getFunction();
    auto Var = declVarInternal<T>("arg." + std::to_string(ArgIdx));
    Var.storeValue(F->getArg(ArgIdx), VarStorage::AccessKind::Direct);
    return Var;
  }

  template <std::size_t... Is> void declArgsImpl(std::index_sequence<Is...>) {
    Function *F = getFunction();
    auto &EntryBB = F->getEntryBlock();
    IP = IRBuilderBase::InsertPoint(&EntryBB, EntryBB.end());
    IRB.restoreIP(IP);

    (std::get<Is>(ArgumentsTT).emplace(createArg<ArgT, Is>()), ...);

    IRB.ClearInsertionPoint();
  }

  template <std::size_t... Is> auto getArgsImpl(std::index_sequence<Is...>) {
    return std::tie(*std::get<Is>(ArgumentsTT)...);
  }

public:
  Func(JitModule &J, FunctionCallee FC, Dispatcher &Dispatch)
      : FuncBase(J, FC), Dispatch(Dispatch) {}

  RetT operator()(ArgT... Args);

  void declArgs() { declArgsImpl(std::index_sequence_for<ArgT...>{}); }

  auto getArgs() { return getArgsImpl(std::index_sequence_for<ArgT...>{}); }

  template <std::size_t Idx> auto &getArg() {
    return *std::get<Idx>(ArgumentsTT);
  }

  auto getCompiledFunc() const { return CompiledFunc; }

  void setCompiledFunc(RetT (*CompiledFuncIn)(ArgT...)) {
    CompiledFunc = CompiledFuncIn;
  }
};

// beginFor implementation
template <typename T>
void FuncBase::beginFor(Var<T> &IterVar, const Var<T> &Init,
                        const Var<T> &UpperBound, const Var<T> &Inc,
                        const char *File, int Line) {
  static_assert(std::is_integral_v<T>,
                "Loop iterator must be an integral type");

  Function *F = getFunction();
  // Update the terminator of the current basic block due to the split
  // control-flow.
  BasicBlock *CurBlock = IP.getBlock();
  BasicBlock *NextBlock =
      CurBlock->splitBasicBlock(IP.getPoint(), CurBlock->getName() + ".split");

  auto ContIP = IRBuilderBase::InsertPoint(NextBlock, NextBlock->begin());
  Scopes.emplace_back(File, Line, ScopeKind::FOR, ContIP);

  BasicBlock *Header =
      BasicBlock::Create(F->getContext(), "loop.header", F, NextBlock);
  BasicBlock *LoopCond =
      BasicBlock::Create(F->getContext(), "loop.cond", F, NextBlock);
  BasicBlock *Body =
      BasicBlock::Create(F->getContext(), "loop.body", F, NextBlock);
  BasicBlock *Latch =
      BasicBlock::Create(F->getContext(), "loop.inc", F, NextBlock);
  BasicBlock *LoopExit =
      BasicBlock::Create(F->getContext(), "loop.end", F, NextBlock);

  // Erase the old terminator and branch to the header.
  CurBlock->getTerminator()->eraseFromParent();
  IRB.SetInsertPoint(CurBlock);
  { IRB.CreateBr(Header); }

  IRB.SetInsertPoint(Header);
  {
    IterVar = Init;
    IRB.CreateBr(LoopCond);
  }

  IRB.SetInsertPoint(LoopCond);
  {
    auto CondVar = IterVar < UpperBound;
    Value *Cond = CondVar.loadValue();
    IRB.CreateCondBr(Cond, Body, LoopExit);
  }

  IRB.SetInsertPoint(Body);
  IRB.CreateBr(Latch);

  IRB.SetInsertPoint(Latch);
  {
    IterVar = IterVar + Inc;
    IRB.CreateBr(LoopCond);
  }

  IRB.SetInsertPoint(LoopExit);
  { IRB.CreateBr(NextBlock); }

  IP = IRBuilderBase::InsertPoint(Body, Body->begin());
  IRB.restoreIP(IP);
}

// Var implementations (defined here after FuncBase is
// complete) so we have it available.

// Helper function for binary operations on Var types
template <typename T, typename U, typename IntOp, typename FPOp>
Var<std::common_type_t<T, U>> binOp(const Var<T> &L, const Var<U> &R, IntOp IOp,
                                    FPOp FOp) {
  FuncBase &Fn = L.Fn;
  if (&Fn != &R.Fn)
    PROTEUS_FATAL_ERROR("Variables should belong to the same function");

  auto &IRB = Fn.getIRBuilder();

  Value *LHS = L.loadValue();
  Value *RHS = R.loadValue();

  using CommonT = std::common_type_t<T, U>;
  LHS = convert<T, CommonT>(IRB, LHS);
  RHS = convert<U, CommonT>(IRB, RHS);

  Value *Result = nullptr;
  if constexpr (std::is_integral_v<CommonT>) {
    Result = IOp(IRB, LHS, RHS);
  } else {
    Result = FOp(IRB, LHS, RHS);
  }

  auto ResultVar = Fn.declVarInternal<std::common_type_t<T, U>>("res.");
  ResultVar.storeValue(Result);

  return ResultVar;
}

// Helper function for compound assignment with a constant
template <typename T, typename U, typename IntOp, typename FPOp>
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>> &
compoundAssignConst(Var<T, std::enable_if_t<std::is_arithmetic_v<T>>> &LHS,
                    const U &ConstValue, IntOp IOp, FPOp FOp) {
  static_assert(std::is_convertible_v<U, T>, "U must be convertible to T");
  auto &IRB = LHS.Fn.getIRBuilder();

  using CleanU = std::remove_cv_t<std::remove_reference_t<U>>;

  Function *Function = LHS.Fn.getFunction();
  auto &Ctx = Function->getContext();
  Type *RHSType = TypeMap<CleanU>::get(Ctx);

  Value *RHS = nullptr;
  if constexpr (std::is_integral_v<CleanU>) {
    RHS = ConstantInt::get(RHSType, ConstValue);
  } else {
    RHS = ConstantFP::get(RHSType, ConstValue);
  }

  Value *LHSVal = LHS.loadValue();

  RHS = convert<CleanU, T>(IRB, RHS);
  Value *Result = nullptr;

  if constexpr (std::is_integral_v<T>) {
    Result = IOp(IRB, LHSVal, RHS);
  } else {
    static_assert(std::is_floating_point_v<T>, "Unsupported type");
    Result = FOp(IRB, LHSVal, RHS);
  }

  LHS.storeValue(Result);
  return LHS;
}

// Helper function for comparison operations on Var types
template <typename T, typename U, typename IntOp, typename FPOp>
Var<bool> cmpOp(const Var<T> &L, const Var<U> &R, IntOp IOp, FPOp FOp) {
  FuncBase &Fn = L.Fn;
  if (&Fn != &R.Fn)
    PROTEUS_FATAL_ERROR("Variables should belong to the same function");

  auto &IRB = Fn.getIRBuilder();

  Value *LHS = L.loadValue();
  Value *RHS = R.loadValue();

  RHS = convert<U, T>(IRB, RHS);

  Value *Result = nullptr;
  if constexpr (std::is_integral_v<T>) {
    Result = IOp(IRB, LHS, RHS);
  } else {
    static_assert(std::is_floating_point_v<T>, "Unsupported type");
    Result = FOp(IRB, LHS, RHS);
  }

  auto ResultVar = Fn.declVarInternal<bool>("res.");
  ResultVar.storeValue(Result);

  return ResultVar;
}

template <typename T>
template <typename U>
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>>::Var(const Var<U> &V)
    : VarStorageOwner<VarStorage>(V.Fn) {
  // Allocate storage for the target type T.
  Type *TargetTy = TypeMap<T>::get(Fn.getFunction()->getContext());
  auto *Alloca = Fn.emitAlloca(TargetTy, "conv.var");
  Storage = std::make_unique<ScalarStorage>(Alloca, Fn.getIRBuilder());
  *this = V;
}

template <typename T>
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>> &
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator=(const Var &V) {
  storeValue(V.loadValue());
  return *this;
}

template <typename T>
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>> &
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator=(Var &&V) {
  if (this->Storage == nullptr) {
    // If we don't have storage, steal it from the source.
    Storage = std::move(V.Storage);
  } else {
    // If we have storage, copy the value.
    storeValue(V.loadValue());
  }
  return *this;
}

template <typename T>
template <typename U>
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>> &
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator=(const Var<U> &V) {
  auto &IRB = Fn.getIRBuilder();
  auto *Converted = convert<U, T>(IRB, V.loadValue());
  storeValue(Converted);
  return *this;
}

template <typename T>
template <typename U>
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>> &
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator=(
    const U &ConstValue) {
  static_assert(std::is_arithmetic_v<U>,
                "Can only assign arithmetic types to Var");

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

template <typename T>
template <typename U>
Var<std::common_type_t<T, U>>
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator+(
    const Var<U> &Other) const {
  return binOp(
      *this, Other,
      [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateAdd(L, R); },
      [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateFAdd(L, R); });
}

template <typename T>
template <typename U>
Var<std::common_type_t<T, U>>
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator-(
    const Var<U> &Other) const {
  return binOp(
      *this, Other,
      [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateSub(L, R); },
      [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateFSub(L, R); });
}

template <typename T>
template <typename U>
Var<std::common_type_t<T, U>>
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator*(
    const Var<U> &Other) const {
  return binOp(
      *this, Other,
      [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateMul(L, R); },
      [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateFMul(L, R); });
}

template <typename T>
template <typename U>
Var<std::common_type_t<T, U>>
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator/(
    const Var<U> &Other) const {
  return binOp(
      *this, Other,
      [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateSDiv(L, R); },
      [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateFDiv(L, R); });
}

template <typename T>
template <typename U>
Var<std::common_type_t<T, U>>
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator%(
    const Var<U> &Other) const {
  return binOp(
      *this, Other,
      [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateSRem(L, R); },
      [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateFRem(L, R); });
}

// Arithmetic operators with ConstValue
template <typename T>
template <typename U, typename>
Var<std::common_type_t<T, U>>
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator+(
    const U &ConstValue) const {
  static_assert(std::is_arithmetic_v<U>,
                "Can only add arithmetic types to Var");
  Var<U> Tmp = Fn.defVar<U>(ConstValue, "tmp.");
  return (*this) + Tmp;
}

template <typename T>
template <typename U, typename>
Var<std::common_type_t<T, U>>
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator-(
    const U &ConstValue) const {
  static_assert(std::is_arithmetic_v<U>,
                "Can only subtract arithmetic types from Var");
  Var<U> Tmp = Fn.defVar<U>(ConstValue, "tmp.");
  return (*this) - Tmp;
}

template <typename T>
template <typename U, typename>
Var<std::common_type_t<T, U>>
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator*(
    const U &ConstValue) const {
  static_assert(std::is_arithmetic_v<U>,
                "Can only multiply Var by arithmetic types");
  Var<U> Tmp = Fn.defVar<U>(ConstValue, "tmp.");
  return (*this) * Tmp;
}

template <typename T>
template <typename U, typename>
Var<std::common_type_t<T, U>>
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator/(
    const U &ConstValue) const {
  static_assert(std::is_arithmetic_v<U>,
                "Can only divide Var by arithmetic types");
  Var<U> Tmp = Fn.defVar<U>(ConstValue, "tmp.");
  return (*this) / Tmp;
}

template <typename T>
template <typename U, typename>
Var<std::common_type_t<T, U>>
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator%(
    const U &ConstValue) const {
  static_assert(std::is_arithmetic_v<U>,
                "Can only modulo Var by arithmetic types");
  Var<U> Tmp = Fn.defVar<U>(ConstValue, "tmp.");
  return (*this) % Tmp;
}

// Compound assignment operators for Var
template <typename T>
template <typename U>
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>> &
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator+=(
    const Var<U> &Other) {
  auto Result = (*this) + Other;
  *this = Result;
  return *this;
}

template <typename T>
template <typename U>
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>> &
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator+=(
    const U &ConstValue) {
  static_assert(std::is_arithmetic_v<U>,
                "Can only add arithmetic types to Var");
  return compoundAssignConst(
      *this, ConstValue,
      [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateAdd(L, R); },
      [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateFAdd(L, R); });
}

template <typename T>
template <typename U>
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>> &
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator-=(
    const Var<U> &Other) {
  auto Result = (*this) - Other;
  *this = Result;
  return *this;
}

template <typename T>
template <typename U>
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>> &
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator-=(
    const U &ConstValue) {
  static_assert(std::is_arithmetic_v<U>,
                "Can only subtract arithmetic types from Var");
  return compoundAssignConst(
      *this, ConstValue,
      [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateSub(L, R); },
      [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateFSub(L, R); });
}

template <typename T>
template <typename U>
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>> &
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator*=(
    const Var<U> &Other) {
  auto Result = (*this) * Other;
  *this = Result;
  return *this;
}

template <typename T>
template <typename U>
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>> &
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator*=(
    const U &ConstValue) {
  static_assert(std::is_arithmetic_v<U>,
                "Can only multiply Var by arithmetic types");
  return compoundAssignConst(
      *this, ConstValue,
      [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateMul(L, R); },
      [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateFMul(L, R); });
}

template <typename T>
template <typename U>
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>> &
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator/=(
    const Var<U> &Other) {
  auto Result = (*this) / Other;
  *this = Result;
  return *this;
}

template <typename T>
template <typename U>
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>> &
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator/=(
    const U &ConstValue) {
  static_assert(std::is_arithmetic_v<U>,
                "Can only divide Var by arithmetic types");
  return compoundAssignConst(
      *this, ConstValue,
      [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateSDiv(L, R); },
      [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateFDiv(L, R); });
}

template <typename T>
template <typename U>
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>> &
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator%=(
    const Var<U> &Other) {
  auto Result = (*this) % Other;
  *this = Result;
  return *this;
}

template <typename T>
template <typename U>
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>> &
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator%=(
    const U &ConstValue) {
  static_assert(std::is_arithmetic_v<U>,
                "Can only modulo Var by arithmetic types");
  return compoundAssignConst(
      *this, ConstValue,
      [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateSRem(L, R); },
      [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateFRem(L, R); });
}

template <typename T>
Var<std::remove_extent_t<T>>
Var<T, std::enable_if_t<std::is_array_v<T>>>::operator[](size_t Index) {
  auto &IRB = Fn.getIRBuilder();
  auto *ArrayTy = cast<ArrayType>(getAllocatedType());
  auto *BasePointer = getSlot();

  // GEP into the array aggregate: [0, Index]
  auto *GEP = IRB.CreateConstInBoundsGEP2_64(ArrayTy, BasePointer, 0, Index);
  Type *ElemTy = getValueType();
  auto *BasePtrTy = cast<PointerType>(BasePointer->getType());
  unsigned AddrSpace = BasePtrTy->getAddressSpace();
  Type *ElemPtrTy = PointerType::get(ElemTy, AddrSpace);

  auto *PtrSlot = Fn.emitAlloca(ElemPtrTy, "elem.ptr");
  IRB.CreateStore(GEP, PtrSlot);

  std::unique_ptr<VarStorage> ResultStorage =
      std::make_unique<PointerStorage>(PtrSlot, IRB, ElemTy);
  return Var<std::remove_extent_t<T>>(std::move(ResultStorage), Fn);
}

template <typename T>
template <typename IdxT>
std::enable_if_t<std::is_integral_v<IdxT>, Var<std::remove_extent_t<T>>>
Var<T, std::enable_if_t<std::is_array_v<T>>>::operator[](
    const Var<IdxT> &Index) {
  auto &IRB = Fn.getIRBuilder();
  auto *ArrayTy = cast<ArrayType>(getAllocatedType());
  auto *BasePointer = getSlot();

  Value *IdxVal = Index.loadValue();
  Value *Zero = llvm::ConstantInt::get(IdxVal->getType(), 0);
  auto *GEP = IRB.CreateInBoundsGEP(ArrayTy, BasePointer, {Zero, IdxVal});
  Type *ElemTy = getValueType();
  auto *BasePtrTy = cast<PointerType>(BasePointer->getType());
  unsigned AddrSpace = BasePtrTy->getAddressSpace();
  Type *ElemPtrTy = PointerType::get(ElemTy, AddrSpace);

  auto *PtrSlot = Fn.emitAlloca(ElemPtrTy, "elem.ptr");
  IRB.CreateStore(GEP, PtrSlot);

  std::unique_ptr<VarStorage> ResultStorage =
      std::make_unique<PointerStorage>(PtrSlot, IRB, ElemTy);
  return Var<std::remove_extent_t<T>>(std::move(ResultStorage), Fn);
}

template <typename T>
Var<std::remove_pointer_t<T>>
Var<T, std::enable_if_t<std::is_pointer_v<T>>>::operator[](size_t Index) {
  auto &IRB = Fn.getIRBuilder();

  auto *PointerElemTy = getValueType();
  auto *Ptr = loadValue(VarStorage::AccessKind::Direct);
  auto *GEP = IRB.CreateConstInBoundsGEP1_64(PointerElemTy, Ptr, Index);
  unsigned AddrSpace = cast<PointerType>(Ptr->getType())->getAddressSpace();
  Type *ElemPtrTy = PointerType::get(PointerElemTy, AddrSpace);

  // Create a pointer storage to hold the LValue for
  // the Array[Index].
  auto *PtrSlot = Fn.emitAlloca(ElemPtrTy, "elem.ptr");
  IRB.CreateStore(GEP, PtrSlot);
  std::unique_ptr<VarStorage> ResultStorage =
      std::make_unique<PointerStorage>(PtrSlot, IRB, PointerElemTy);

  return Var<std::remove_pointer_t<T>>(std::move(ResultStorage), Fn);
}

template <typename T>
template <typename IdxT>
std::enable_if_t<std::is_arithmetic_v<IdxT>, Var<std::remove_pointer_t<T>>>
Var<T, std::enable_if_t<std::is_pointer_v<T>>>::operator[](
    const Var<IdxT> &Index) {
  auto &IRB = Fn.getIRBuilder();

  auto *PointeeType = getValueType();
  auto *Ptr = loadValue(VarStorage::AccessKind::Direct);
  auto *IdxValue = Index.loadValue();
  auto *GEP = IRB.CreateInBoundsGEP(PointeeType, Ptr, IdxValue);
  unsigned AddrSpace = cast<PointerType>(Ptr->getType())->getAddressSpace();
  Type *ElemPtrTy = PointerType::get(PointeeType, AddrSpace);

  // Create a pointer storage to hold the LValue for
  // the Array[Index].
  auto *PtrSlot = Fn.emitAlloca(ElemPtrTy, "elem.ptr");
  IRB.CreateStore(GEP, PtrSlot);
  std::unique_ptr<VarStorage> ResultStorage =
      std::make_unique<PointerStorage>(PtrSlot, IRB, PointeeType);

  return Var<std::remove_pointer_t<T>>(std::move(ResultStorage), Fn);
}

// Pointer type operator*
template <typename T>
Var<std::remove_pointer_t<T>>
Var<T, std::enable_if_t<std::is_pointer_v<T>>>::operator*() {
  return (*this)[0];
}

template <typename T>
template <typename OffsetT>
std::enable_if_t<std::is_arithmetic_v<OffsetT>,
                 Var<T, std::enable_if_t<std::is_pointer_v<T>>>>
Var<T, std::enable_if_t<std::is_pointer_v<T>>>::operator+(
    const Var<OffsetT> &Offset) const {
  auto &IRB = Fn.getIRBuilder();

  auto *OffsetVal = Offset.loadValue();
  auto *IdxVal = convert<OffsetT, int64_t>(IRB, OffsetVal);

  auto *BasePtr = loadValue(VarStorage::AccessKind::Direct);
  auto *ElemTy = getValueType();

  auto *GEP = IRB.CreateInBoundsGEP(ElemTy, BasePtr, IdxVal, "ptr.add");

  unsigned AddrSpace = cast<PointerType>(BasePtr->getType())->getAddressSpace();
  auto *ElemPtrTy = PointerType::get(ElemTy, AddrSpace);
  auto *PtrSlot = Fn.emitAlloca(ElemPtrTy, "ptr.add.tmp");
  IRB.CreateStore(GEP, PtrSlot);
  std::unique_ptr<PointerStorage> ResultStorage =
      std::make_unique<PointerStorage>(PtrSlot, IRB, ElemTy);
  return Var<T, std::enable_if_t<std::is_pointer_v<T>>>(
      std::move(ResultStorage), Fn);
}

template <typename T>
template <typename OffsetT>
std::enable_if_t<std::is_arithmetic_v<OffsetT>,
                 Var<T, std::enable_if_t<std::is_pointer_v<T>>>>
Var<T, std::enable_if_t<std::is_pointer_v<T>>>::operator+(
    OffsetT Offset) const {
  auto &IRB = Fn.getIRBuilder();
  auto *IntTy = IRB.getInt64Ty();
  Value *IdxVal = ConstantInt::get(IntTy, Offset);

  auto *BasePtr = loadValue(VarStorage::AccessKind::Direct);
  auto *ElemTy = getValueType();

  auto *GEP = IRB.CreateInBoundsGEP(ElemTy, BasePtr, IdxVal, "ptr.add");

  unsigned AddrSpace = cast<PointerType>(BasePtr->getType())->getAddressSpace();
  auto *ElemPtrTy = PointerType::get(ElemTy, AddrSpace);
  auto *PtrSlot = Fn.emitAlloca(ElemPtrTy, "ptr.add.tmp");
  IRB.CreateStore(GEP, PtrSlot);
  std::unique_ptr<PointerStorage> ResultStorage =
      std::make_unique<PointerStorage>(PtrSlot, IRB, ElemTy);
  return Var<T, std::enable_if_t<std::is_pointer_v<T>>>(
      std::move(ResultStorage), Fn);
}

// Comparison operators for Var
template <typename T>
template <typename U>
std::enable_if_t<std::is_arithmetic_v<U>, Var<bool>>
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator>(
    const Var<U> &Other) const {
  return cmpOp(
      *this, Other,
      [](IRBuilderBase &B, Value *L, Value *R) {
        return B.CreateICmpSGT(L, R);
      },
      [](IRBuilderBase &B, Value *L, Value *R) {
        return B.CreateFCmpOGT(L, R);
      });
}

template <typename T>
template <typename U>
std::enable_if_t<std::is_arithmetic_v<U>, Var<bool>>
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator>=(
    const Var<U> &Other) const {
  return cmpOp(
      *this, Other,
      [](IRBuilderBase &B, Value *L, Value *R) {
        return B.CreateICmpSGE(L, R);
      },
      [](IRBuilderBase &B, Value *L, Value *R) {
        return B.CreateFCmpOGE(L, R);
      });
}

template <typename T>
template <typename U>
std::enable_if_t<std::is_arithmetic_v<U>, Var<bool>>
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator<(
    const Var<U> &Other) const {
  return cmpOp(
      *this, Other,
      [](IRBuilderBase &B, Value *L, Value *R) {
        return B.CreateICmpSLT(L, R);
      },
      [](IRBuilderBase &B, Value *L, Value *R) {
        return B.CreateFCmpOLT(L, R);
      });
}

template <typename T>
template <typename U>
std::enable_if_t<std::is_arithmetic_v<U>, Var<bool>>
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator<=(
    const Var<U> &Other) const {
  return cmpOp(
      *this, Other,
      [](IRBuilderBase &B, Value *L, Value *R) {
        return B.CreateICmpSLE(L, R);
      },
      [](IRBuilderBase &B, Value *L, Value *R) {
        return B.CreateFCmpOLE(L, R);
      });
}

template <typename T>
template <typename U>
std::enable_if_t<std::is_arithmetic_v<U>, Var<bool>>
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator==(
    const Var<U> &Other) const {
  return cmpOp(
      *this, Other,
      [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateICmpEQ(L, R); },
      [](IRBuilderBase &B, Value *L, Value *R) {
        return B.CreateFCmpOEQ(L, R);
      });
}

template <typename T>
template <typename U>
std::enable_if_t<std::is_arithmetic_v<U>, Var<bool>>
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator!=(
    const Var<U> &Other) const {
  return cmpOp(
      *this, Other,
      [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateICmpNE(L, R); },
      [](IRBuilderBase &B, Value *L, Value *R) {
        return B.CreateFCmpONE(L, R);
      });
}

template <typename T>
template <typename U>
std::enable_if_t<std::is_arithmetic_v<U>, Var<bool>>
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator>(
    const U &ConstValue) const {
  Var<U> Tmp = Fn.defVar<U>(ConstValue, "cmp.");
  return (*this) > Tmp;
}

template <typename T>
template <typename U>
std::enable_if_t<std::is_arithmetic_v<U>, Var<bool>>
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator>=(
    const U &ConstValue) const {
  Var<U> Tmp = Fn.defVar<U>(ConstValue, "cmp.");
  return (*this) >= Tmp;
}

template <typename T>
template <typename U>
std::enable_if_t<std::is_arithmetic_v<U>, Var<bool>>
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator<(
    const U &ConstValue) const {
  Var<U> Tmp = Fn.defVar<U>(ConstValue, "cmp.");
  return (*this) < Tmp;
}

template <typename T>
template <typename U>
std::enable_if_t<std::is_arithmetic_v<U>, Var<bool>>
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator<=(
    const U &ConstValue) const {
  auto Tmp = Fn.defVar<U>(ConstValue, "cmp.");
  return (*this) <= Tmp;
}

template <typename T>
template <typename U>
std::enable_if_t<std::is_arithmetic_v<U>, Var<bool>>
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator==(
    const U &ConstValue) const {
  Var<U> Tmp = Fn.defVar<U>(ConstValue, "cmp.");
  return (*this) == Tmp;
}

template <typename T>
template <typename U>
std::enable_if_t<std::is_arithmetic_v<U>, Var<bool>>
Var<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator!=(
    const U &ConstValue) const {
  auto Tmp = Fn.defVar<U>(ConstValue, "cmp.");
  return (*this) != Tmp;
}

// Non-member arithmetic operators for Var
template <typename T, typename U>
std::enable_if_t<std::is_arithmetic_v<T> && std::is_arithmetic_v<U>,
                 Var<std::common_type_t<T, U>>>
operator+(const T &ConstValue, const Var<U> &V) {
  Var<T> Tmp = V.Fn.template defVar<T>(ConstValue, "tmp.");
  return Tmp + V;
}

template <typename T, typename U>
std::enable_if_t<std::is_arithmetic_v<T> && std::is_arithmetic_v<U>,
                 Var<std::common_type_t<T, U>>>
operator-(const T &ConstValue, const Var<U> &V) {
  using CommonType = std::common_type_t<T, U>;
  Var<CommonType> Tmp = V.Fn.template defVar<CommonType>(ConstValue, "tmp.");
  return Tmp - V;
}

template <typename T, typename U>
std::enable_if_t<std::is_arithmetic_v<T> && std::is_arithmetic_v<U>,
                 Var<std::common_type_t<T, U>>>
operator*(const T &ConstValue, const Var<U> &V) {
  Var<T> Tmp = V.Fn.template defVar<T>(ConstValue, "tmp.");
  return Tmp * V;
}

template <typename T, typename U>
std::enable_if_t<std::is_arithmetic_v<T> && std::is_arithmetic_v<U>,
                 Var<std::common_type_t<T, U>>>
operator/(const T &ConstValue, const Var<U> &V) {
  Var<T> Tmp = V.Fn.template defVar<T>(ConstValue, "tmp.");
  return Tmp / V;
}

template <typename T, typename U>
std::enable_if_t<std::is_arithmetic_v<T> && std::is_arithmetic_v<U>,
                 Var<std::common_type_t<T, U>>>
operator%(const T &ConstValue, const Var<U> &V) {
  Var<T> Tmp = V.Fn.template defVar<T>(ConstValue, "tmp.");
  return Tmp % V;
}

// Atomic operations for Var
template <typename T>
Var<T> FuncBase::emitAtomic(AtomicRMWInst::BinOp Op, const Var<T *> &Addr,
                            const Var<T> &Val) {
  static_assert(std::is_arithmetic_v<T>, "Atomic ops require arithmetic type");

  auto &IRB = getIRBuilder();
  auto *Result = IRB.CreateAtomicRMW(
      Op, Addr.loadValue(VarStorage::AccessKind::Direct), Val.loadValue(),
      MaybeAlign(), AtomicOrdering::SequentiallyConsistent,
      SyncScope::SingleThread);

  auto Ret = declVarInternal<T>("atomic.rmw.res.");
  Ret.storeValue(Result);
  return Ret;
}

template <typename T>
std::enable_if_t<std::is_arithmetic_v<T>, Var<T>>
FuncBase::atomicAdd(const Var<T *> &Addr, const Var<T> &Val) {
  static_assert(std::is_arithmetic_v<T>, "atomicAdd requires arithmetic type");

  Type *ValueType = TypeMap<T>::get(getFunction()->getContext());
  auto Op =
      ValueType->isFloatingPointTy() ? AtomicRMWInst::FAdd : AtomicRMWInst::Add;
  return emitAtomic(Op, Addr, Val);
}

template <typename T>
std::enable_if_t<std::is_arithmetic_v<T>, Var<T>>
FuncBase::atomicSub(const Var<T *> &Addr, const Var<T> &Val) {
  static_assert(std::is_arithmetic_v<T>, "atomicSub requires arithmetic type");

  Type *ValueType = TypeMap<T>::get(getFunction()->getContext());
  auto Op =
      ValueType->isFloatingPointTy() ? AtomicRMWInst::FSub : AtomicRMWInst::Sub;
  return emitAtomic(Op, Addr, Val);
}

template <typename T>
std::enable_if_t<std::is_arithmetic_v<T>, Var<T>>
FuncBase::atomicMax(const Var<T *> &Addr, const Var<T> &Val) {
  static_assert(std::is_arithmetic_v<T>, "atomicMax requires arithmetic type");

  Type *ValueType = TypeMap<T>::get(getFunction()->getContext());
  auto Op =
      ValueType->isFloatingPointTy() ? AtomicRMWInst::FMax : AtomicRMWInst::Max;
  return emitAtomic(Op, Addr, Val);
}

template <typename T>
std::enable_if_t<std::is_arithmetic_v<T>, Var<T>>
FuncBase::atomicMin(const Var<T *> &Addr, const Var<T> &Val) {
  static_assert(std::is_arithmetic_v<T>, "atomicMin requires arithmetic type");

  Type *ValueType = TypeMap<T>::get(getFunction()->getContext());
  auto Op =
      ValueType->isFloatingPointTy() ? AtomicRMWInst::FMin : AtomicRMWInst::Min;
  return emitAtomic(Op, Addr, Val);
}

inline void FuncBase::ret() {
  auto *CurBB = IP.getBlock();
  if (!CurBB->getSingleSuccessor())
    PROTEUS_FATAL_ERROR("Expected single successor for current block");
  auto *TermI = CurBB->getTerminator();

  auto &IRB = getIRBuilder();
  IRB.CreateRetVoid();

  TermI->eraseFromParent();
}

template <typename T> void FuncBase::ret(const Var<T> &RetVal) {
  auto *CurBB = IP.getBlock();
  if (!CurBB->getSingleSuccessor())
    PROTEUS_FATAL_ERROR("Expected single successor for current block");
  auto *TermI = CurBB->getTerminator();

  auto &IRB = getIRBuilder();
  Value *RetValue = RetVal.loadValue();
  IRB.CreateRet(RetValue);

  TermI->eraseFromParent();
}

// Helper struct to convert Var operands to a target type T.
// Used by emitIntrinsic to convert all operands to the intrinsic's result type.
// C++17 doesn't support template parameters on lambdas, so we use a struct.
template <typename T> struct IntrinsicOperandConverter {
  IRBuilderBase &IRB;

  template <typename U> Value *operator()(const Var<U> &Operand) const {
    return convert<U, T>(IRB, Operand.loadValue());
  }
};

// Helper for emitting intrinsics with Var
template <typename T, typename... Operands>
static Var<T> emitIntrinsic(StringRef IntrinsicName, Type *ResultType,
                            const Operands &...Ops) {
  static_assert(sizeof...(Ops) > 0, "Intrinsic requires at least one operand");

  auto &Fn = std::get<0>(std::tie(Ops...)).Fn;
  auto CheckFn = [&Fn](const auto &Operand) {
    if (&Operand.Fn != &Fn)
      PROTEUS_FATAL_ERROR("Variables should belong to the same function");
  };
  (CheckFn(Ops), ...);

  auto &IRB = Fn.getIRBuilder();
  auto &M = *Fn.getFunction()->getParent();

  IntrinsicOperandConverter<T> ConvertOperand{IRB};

  FunctionCallee Callee = M.getOrInsertFunction(IntrinsicName, ResultType,
                                                ((void)Ops, ResultType)...);
  Value *Call = IRB.CreateCall(Callee, {ConvertOperand(Ops)...});

  auto ResultVar = Fn.template declVar<T>("res.");
  ResultVar.storeValue(Call);
  return ResultVar;
}

// Math intrinsics for Var
template <typename T> Var<float> powf(const Var<float> &L, const Var<T> &R) {
  static_assert(std::is_convertible_v<T, float>,
                "powf requires floating-point type");
  auto &IRB = L.Fn.getIRBuilder();
  auto *ResultType = IRB.getFloatTy();
  auto RFloat = R.Fn.template convert<float>(R);

#if PROTEUS_ENABLE_CUDA
  std::string IntrinsicName = "__nv_powf";
#else
  std::string IntrinsicName = "llvm.pow.f32";
#endif

  return emitIntrinsic<float>(IntrinsicName, ResultType, L, RFloat);
}

template <typename T> Var<float> sqrtf(const Var<T> &R) {
  static_assert(std::is_convertible_v<T, float>,
                "sqrtf requires floating-point type");

  auto &IRB = R.Fn.getIRBuilder();
  auto *ResultType = IRB.getFloatTy();
  auto RFloat = R.Fn.template convert<float>(R);

#if PROTEUS_ENABLE_CUDA
  std::string IntrinsicName = "__nv_sqrtf";
#else
  std::string IntrinsicName = "llvm.sqrt.f32";
#endif

  return emitIntrinsic<float>(IntrinsicName, ResultType, RFloat);
}

template <typename T>
std::enable_if_t<std::is_arithmetic_v<T>, Var<T>> min(const Var<T> &L,
                                                      const Var<T> &R) {
  static_assert(std::is_arithmetic_v<T>, "min requires arithmetic type");

  FuncBase &Fn = L.Fn;
  if (&Fn != &R.Fn)
    PROTEUS_FATAL_ERROR("Variables should belong to the same function");

  Var<T> ResultVar = Fn.declVar<T>("min_res");
  ResultVar = R;
  Fn.beginIf(L < R);
  { ResultVar = L; }
  Fn.endIf();
  return ResultVar;
}

} // namespace proteus

#endif // PROTEUS_FRONTEND_FUNC_HPP
