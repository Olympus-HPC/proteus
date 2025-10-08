#ifndef PROTEUS_FRONTEND_FUNC_HPP
#define PROTEUS_FRONTEND_FUNC_HPP

#include <deque>
#include <memory>

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

#include "proteus/AddressSpace.hpp"
#include "proteus/Error.h"
#include "proteus/Frontend/Dispatcher.hpp"
#include "proteus/Frontend/TypeMap.hpp"
#include "proteus/Frontend/VarStorage.hpp"
#include "proteus/Frontend/Var.hpp"

namespace proteus {

struct Var;
class JitModule;
class LoopBoundInfo;
template <typename... ForLoopBuilders> class LoopNestBuilder;
template <typename BodyLambda> class ForLoopBuilder;

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

  std::deque<std::unique_ptr<Var>> Arguments;
  std::deque<std::unique_ptr<Var>> Variables;
  std::deque<std::unique_ptr<Var>> RuntimeConstants;

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

  Var &emitAtomic(AtomicRMWInst::BinOp Op, Var &Addr, Var &Val);

public:
  FuncBase(JitModule &J, FunctionCallee FC);

  Function *getFunction();

  AllocaInst *emitAlloca(Type *Ty, StringRef Name,
                         AddressSpace AS = AddressSpace::DEFAULT);

  Value *emitArrayCreate(Type *Ty, AddressSpace AT, StringRef Name);

  IRBuilderBase &getIRBuilder();

  // Create a variable. If PointerElemType is non-null, a PointerVar is created
  // with an alloca-backed pointer slot; otherwise a ScalarVar is created.
  Var &declVarInternal(StringRef Name, Type *Ty,
                       Type *PointerElemType = nullptr);

  template <typename T> 
  VarTT<T> declVarTTInternal(StringRef Name = "var") {
    static_assert(!std::is_array_v<T>, "Expected non-array type");
    
    Function *F = getFunction();
    auto &Ctx = F->getContext();
    Type *AllocaTy = TypeMap<T>::get(Ctx);
    auto *Alloca = emitAlloca(AllocaTy, Name);
    
    if constexpr (std::is_pointer_v<T>) {
      Type *PtrElemTy = TypeMap<T>::getPointerElemType(Ctx);
      return VarTT<T>(std::make_unique<PointerStorage>(Alloca, IRB, PtrElemTy), *this);
    } else {
      return VarTT<T>(std::make_unique<ScalarStorage>(Alloca, IRB), *this);
    }
  }

  template <typename T> Var &declVar(StringRef Name = "var") {
    static_assert(!std::is_array_v<T>, "Expected non-array type");

    Function *F = getFunction();
    return declVarInternal(Name, TypeMap<T>::get(F->getContext()));
  }

  template <typename T>
  Var &declVar(size_t NElem, AddressSpace AS = AddressSpace::DEFAULT,
               StringRef Name = "array_var") {
    static_assert(std::is_array_v<T>, "Expected array type");

    Function *F = getFunction();
    auto *BasePointer =
        emitArrayCreate(TypeMap<T>::get(F->getContext(), NElem), AS, Name);

    auto *ArrTy = cast<ArrayType>(TypeMap<T>::get(F->getContext(), NElem));
    auto &Ref = *Variables.emplace_back(
        std::make_unique<ArrayVar>(BasePointer, *this, ArrTy));
    return Ref;
  }

  template <typename T> Var &defVar(T Val, StringRef Name = "var") {
    Function *F = getFunction();
    Var &VarRef = declVarInternal(Name, TypeMap<T>::get(F->getContext()));
    VarRef = Val;
    return VarRef;
  }

  Var &defVar(const Var &Val, StringRef Name = "var") {
    if (&Val.Fn != this)
      PROTEUS_FATAL_ERROR("Variables should belong to the same function");
    Var &VarRef = declVarInternal(Name, Val.getValueType());
    VarRef = Val;
    return VarRef;
  }

  template <typename T>
  Var &defRuntimeConst(T Val, StringRef Name = "run.const.var") {
    Function *F = getFunction();
    Var &VarRef = declVarInternal(Name, TypeMap<T>::get(F->getContext()));
    VarRef = Val;
    return VarRef;
  }

  template<typename T>
  VarTT<T> declVarTT(StringRef Name = "var") {
    static_assert(!std::is_array_v<T>, "Expected non-array type");
    return declVarTTInternal<T>(Name);
  }

  template <typename T>
  VarTT<T> defVarTT(T Val, StringRef Name = "var") {
    VarTT<T> Var = declVarTTInternal<T>(Name);
    Var = Val;
    return Var;
  }

  template <typename... ArgT> auto defRuntimeConsts(ArgT &&...Args) {
    return std::tie(defRuntimeConst(std::forward<ArgT>(Args))...);
  }

  template <typename... Ts> void declArgs() {
    Function *F = getFunction();
    auto &EntryBB = F->getEntryBlock();
    IP = IRBuilderBase::InsertPoint(&EntryBB, EntryBB.end());
    IRB.restoreIP(IP);

    (
        [&]() {
          // Determine if this argument is a pointer-like type.
          auto &Ctx = F->getContext();
          Type *ArgTy = TypeMap<Ts>::get(Ctx);
          Type *PtrElemTy = TypeMap<Ts>::getPointerElemType(Ctx);

          // Allocate a slot to hold the incoming argument value (pointer or
          // scalar).
          auto *Alloca =
              emitAlloca(ArgTy, "arg." + std::to_string(Arguments.size()));

          auto *Arg = F->getArg(Arguments.size());
          IRB.CreateStore(Arg, Alloca);

          if (PtrElemTy) {
            // Pointer argument: create a PointerVar with the correct element
            // type.
            Arguments.emplace_back(
                std::make_unique<PointerVar>(Alloca, *this, PtrElemTy));
          } else {
            // Scalar argument.
            Arguments.emplace_back(std::make_unique<ScalarVar>(Alloca, *this));
          }
        }(),
        ...);
    IRB.ClearInsertionPoint();
  }

  Var &getArg(unsigned int ArgNo);

  void beginFunction(const char *File = __builtin_FILE(),
                     int Line = __builtin_LINE());
  void endFunction();

  void beginIf(Var &CondVar, const char *File = __builtin_FILE(),
               int Line = __builtin_LINE());
  void endIf();

  void beginIfTT(const VarTT<bool> &CondVar, const char *File = __builtin_FILE(),
                 int Line = __builtin_LINE());
  void endIfTT();

  void beginFor(Var &IterVar, Var &InitVar, Var &UpperBound, Var &IncVar,
                const char *File = __builtin_FILE(),
                int Line = __builtin_LINE());
  void endFor();

  template <typename Sig>
  std::enable_if_t<!std::is_void_v<typename FnSig<Sig>::RetT>, Var &>
  call(StringRef Name);

  template <typename Sig>
  std::enable_if_t<std::is_void_v<typename FnSig<Sig>::RetT>, void>
  call(StringRef Name);

  template <typename Sig, typename... ArgVars>
  std::enable_if_t<!std::is_void_v<typename FnSig<Sig>::RetT>, Var &>
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

  Var &atomicAdd(Var &Addr, Var &Val);
  Var &atomicSub(Var &Addr, Var &Val);
  Var &atomicMax(Var &Addr, Var &Val);
  Var &atomicMin(Var &Addr, Var &Val);

  template <typename BodyLambda = EmptyLambda>
  auto forLoop(const LoopBoundInfo &Bounds, BodyLambda &&Body = {}) {
    return ForLoopBuilder(Bounds, *this, std::move(Body));
  }

  template <typename... LoopBuilders>
  auto buildLoopNest(LoopBuilders &&...Loops) {
    return LoopNestBuilder(*this, std::forward<LoopBuilders>(Loops)...);
  }

  void ret(std::optional<std::reference_wrapper<Var>> OptRet = std::nullopt);

  StringRef getName() const { return Name; }

  void setName(StringRef NewName) {
    Name = NewName.str();
    Function *F = getFunction();
    F->setName(Name);
  }

  // Convert the given Var's value to type T and return a new Var holding
  // the converted value.
  template <typename T>
  std::enable_if_t<std::is_arithmetic_v<T>, Var &> convert(Var &V) {
    auto &Ctx = getFunction()->getContext();
    auto &IRBRef = getIRBuilder();
    Type *TargetTy = TypeMap<T>::get(Ctx);
    Var &Res = declVarInternal("convert.", TargetTy);
    Value *Converted = proteus::convert(IRBRef, V.getValue(), TargetTy);
    Res.storeValue(Converted);
    return Res;
  }
};

template <typename RetT, typename... ArgT> class Func final : public FuncBase {
private:
  Dispatcher &Dispatch;
  RetT (*CompiledFunc)(ArgT...) = nullptr;
  // Optional because ArgTT is not default constructible.
  std::tuple<std::optional<VarTT<ArgT>>...> ArgumentsTT;

private:
  template <std::size_t... Is> auto getArgsImpl(std::index_sequence<Is...>) {
    return std::tie(getArg(Is)...);
  }

  template <typename T, std::size_t ArgIdx>
  VarTT<T> createArgTT() {
    Function *F = getFunction();
    auto &Ctx = F->getContext();
    
    // Get the LLVM type for this argument
    Type *ArgTy = TypeMap<T>::get(Ctx);
    
    // Create alloca to hold the incoming argument value
    auto *Alloca = emitAlloca(ArgTy, "arg." + std::to_string(ArgIdx));
    
    // Store the function argument into the alloca
    auto *Arg = F->getArg(ArgIdx);
    IRB.CreateStore(Arg, Alloca);
    
    // Create appropriate storage based on type
    if constexpr (std::is_pointer_v<T>) {
      // Pointer type: use PointerStorage
      Type *PtrElemTy = TypeMap<T>::getPointerElemType(Ctx);
      return VarTT<T>(std::make_unique<PointerStorage>(Alloca, IRB, PtrElemTy), *this);
    } else {
      // Scalar type: use ScalarStorage
      return VarTT<T>(std::make_unique<ScalarStorage>(Alloca, IRB), *this);
    }
  }

  template <std::size_t... Is>
  void declArgsTTImpl(std::index_sequence<Is...>) {
    Function *F = getFunction();
    auto &EntryBB = F->getEntryBlock();
    IP = IRBuilderBase::InsertPoint(&EntryBB, EntryBB.end());
    IRB.restoreIP(IP);
    
    (std::get<Is>(ArgumentsTT).emplace(createArgTT<ArgT, Is>()), ...);
    
    IRB.ClearInsertionPoint();
  }

public:
  Func(JitModule &J, FunctionCallee FC, Dispatcher &Dispatch)
      : FuncBase(J, FC), Dispatch(Dispatch) {}

  RetT operator()(ArgT... Args);

  auto getArgs() { return getArgsImpl(std::index_sequence_for<ArgT...>{}); }
  
  void declArgsTT() {
    declArgsTTImpl(std::index_sequence_for<ArgT...>{});
  }
  
  auto& getArgsTT() {
    return ArgumentsTT;
  }
  
  template<std::size_t Idx>
  auto& getArgTT() {
    return *std::get<Idx>(ArgumentsTT);
  }

  auto getCompiledFunc() const { return CompiledFunc; }

  void setCompiledFunc(RetT (*CompiledFuncIn)(ArgT...)) {
    CompiledFunc = CompiledFuncIn;
  }
};

// VarTT arithmetic specialization implementations (defined here after FuncBase is complete)
// so we have it available.

// Helper function for binary operations on VarTT types
template <typename T, typename U, typename IntOp, typename FPOp>
VarTT<std::common_type_t<T, U>> binOpTT(const VarTT<T> &L, const VarTT<U> &R, IntOp IOp, FPOp FOp) {
  FuncBase &Fn = L.Fn;
  if (&Fn != &R.Fn)
    PROTEUS_FATAL_ERROR("Variables should belong to the same function");
  
  auto &IRB = Fn.getIRBuilder();
  
  Value *LHS = L.Storage->loadValue();
  Value *RHS = R.Storage->loadValue();
  
  Value *Result = nullptr;
  if constexpr (std::is_integral_v<std::common_type_t<T, U>>) {
    Result = IOp(IRB, LHS, RHS);
  } else {
    Result = FOp(IRB, LHS, RHS);
  }
  
  auto *ResultSlot = IRB.CreateAlloca(Result->getType());
  IRB.CreateStore(Result, ResultSlot);
  
  std::unique_ptr<VarStorage> ResultStorage = std::make_unique<ScalarStorage>(ResultSlot, IRB);
  return VarTT<std::common_type_t<T, U>>(std::move(ResultStorage), Fn);
}

// Helper function for compound assignment with another VarTT
template <typename T, typename U, typename BinOp>
VarTT<T, std::enable_if_t<std::is_arithmetic_v<T>>> &
compoundAssignVarTT(VarTT<T, std::enable_if_t<std::is_arithmetic_v<T>>> &LHS,
                    const VarTT<U> &RHS,
                    BinOp Op) {
  auto Result = Op(LHS, RHS);
  auto &IRB = LHS.Fn.getIRBuilder();
  auto *Converted = convert(IRB, Result.Storage->loadValue(), LHS.Storage->getValueType());
  LHS.Storage->storeValue(Converted);
  return LHS;
}

// Helper function for compound assignment with a constant
template <typename T, typename U, typename IntOp, typename FPOp>
VarTT<T, std::enable_if_t<std::is_arithmetic_v<T>>> &
compoundAssignConstTT(VarTT<T, std::enable_if_t<std::is_arithmetic_v<T>>> &LHS,
                      const U &ConstValue,
                      IntOp IOp, FPOp FOp) {
  Type *LHSType = LHS.Storage->getValueType();
  Value *RHS = nullptr;
  
  if constexpr (std::is_integral_v<T>) {
    RHS = ConstantInt::get(LHSType, ConstValue);
  } else {
    RHS = ConstantFP::get(LHSType, ConstValue);
  }
  
  auto &IRB = LHS.Fn.getIRBuilder();
  Value *LHSVal = LHS.Storage->loadValue();
  Value *Result = nullptr;
  
  if constexpr (std::is_integral_v<T>) {
    Result = IOp(IRB, LHSVal, RHS);
  } else {
    Result = FOp(IRB, LHSVal, RHS);
  }
  
  LHS.Storage->storeValue(Result);
  return LHS;
}

// Helper function for comparison operations on VarTT types
template <typename T, typename U, typename IntOp, typename FPOp>
VarTT<bool> cmpOpTT(const VarTT<T> &L, const VarTT<U> &R, IntOp IOp, FPOp FOp) {
  FuncBase &Fn = L.Fn;
  if (&Fn != &R.Fn)
    PROTEUS_FATAL_ERROR("Variables should belong to the same function");
  
  auto &IRB = Fn.getIRBuilder();
  
  Value *LHS = L.Storage->loadValue();
  Value *RHS = R.Storage->loadValue();
  
  Value *Result = nullptr;
  if constexpr (std::is_integral_v<std::common_type_t<T, U>>) {
    Result = IOp(IRB, LHS, RHS);
  } else {
    Result = FOp(IRB, LHS, RHS);
  }
  
  auto *ResultSlot = IRB.CreateAlloca(Result->getType());
  IRB.CreateStore(Result, ResultSlot);
  
  std::unique_ptr<VarStorage> ResultStorage = std::make_unique<ScalarStorage>(ResultSlot, IRB);
  return VarTT<bool>(std::move(ResultStorage), Fn);
}

template <typename T>
VarTT<T, std::enable_if_t<std::is_arithmetic_v<T>>> &
VarTT<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator=(const VarTT &Var) {
  auto &IRB = Fn.getIRBuilder();
  auto *Converted = convert(IRB, Var.Storage->loadValue(), Storage->getValueType());
  Storage->storeValue(Converted);
  return *this;
}

template <typename T>
template <typename U>
VarTT<T, std::enable_if_t<std::is_arithmetic_v<T>>> &
VarTT<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator=(const VarTT<U> &Var) {
  auto &IRB = Fn.getIRBuilder();
  auto *Converted = convert(IRB, Var.Storage->loadValue(), Storage->getValueType());
  Storage->storeValue(Converted);
  return *this;
}

template <typename T>
template <typename U>
VarTT<T, std::enable_if_t<std::is_arithmetic_v<T>>> &
VarTT<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator=(const U &ConstValue) {
  static_assert(std::is_arithmetic_v<U>, "Can only assign arithmetic types to VarTT");
  
  Type *LHSType = Storage->getValueType();
  
  if (LHSType->isIntegerTy()) {
    Storage->storeValue(ConstantInt::get(LHSType, ConstValue));
  } else if (LHSType->isFloatingPointTy()) {
    Storage->storeValue(ConstantFP::get(LHSType, ConstValue));
  } else {
    PROTEUS_FATAL_ERROR("Unsupported type");
  }
  
  return *this;
}

template <typename T>
template <typename U>
VarTT<std::common_type_t<T, U>> VarTT<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator+(
    const VarTT<U> &Other) const {
  return binOpTT(
      *this, Other,
      [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateAdd(L, R); },
      [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateFAdd(L, R); });
}

template <typename T>
template <typename U>
VarTT<std::common_type_t<T, U>> VarTT<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator-(
    const VarTT<U> &Other) const {
  return binOpTT(
      *this, Other,
      [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateSub(L, R); },
      [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateFSub(L, R); });
}

template <typename T>
template <typename U>
VarTT<std::common_type_t<T, U>> VarTT<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator*(
    const VarTT<U> &Other) const {
  return binOpTT(
      *this, Other,
      [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateMul(L, R); },
      [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateFMul(L, R); });
}

template <typename T>
template <typename U>
VarTT<std::common_type_t<T, U>> VarTT<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator/(
    const VarTT<U> &Other) const {
  return binOpTT(
      *this, Other,
      [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateSDiv(L, R); },
      [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateFDiv(L, R); });
}

template <typename T>
template <typename U>
VarTT<std::common_type_t<T, U>> VarTT<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator%(
    const VarTT<U> &Other) const {
  return binOpTT(
      *this, Other,
      [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateSRem(L, R); },
      [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateFRem(L, R); });
}

// Compound assignment operators for VarTT
template <typename T>
template <typename U>
VarTT<T, std::enable_if_t<std::is_arithmetic_v<T>>> &
VarTT<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator+=(const VarTT<U> &Other) {
  return compoundAssignVarTT(*this, Other, 
    [](auto &L, auto &R) { return L + R; });
}

template <typename T>
template <typename U>
VarTT<T, std::enable_if_t<std::is_arithmetic_v<T>>> &
VarTT<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator+=(const U &ConstValue) {
  static_assert(std::is_arithmetic_v<U>, "Can only add arithmetic types to VarTT");
  return compoundAssignConstTT(*this, ConstValue,
    [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateAdd(L, R); },
    [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateFAdd(L, R); });
}

template <typename T>
template <typename U>
VarTT<T, std::enable_if_t<std::is_arithmetic_v<T>>> &
VarTT<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator-=(const VarTT<U> &Other) {
  return compoundAssignVarTT(*this, Other, 
    [](auto &L, auto &R) { return L - R; });
}

template <typename T>
template <typename U>
VarTT<T, std::enable_if_t<std::is_arithmetic_v<T>>> &
VarTT<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator-=(const U &ConstValue) {
  static_assert(std::is_arithmetic_v<U>, "Can only subtract arithmetic types from VarTT");
  return compoundAssignConstTT(*this, ConstValue,
    [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateSub(L, R); },
    [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateFSub(L, R); });
}

template <typename T>
template <typename U>
VarTT<T, std::enable_if_t<std::is_arithmetic_v<T>>> &
VarTT<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator*=(const VarTT<U> &Other) {
  return compoundAssignVarTT(*this, Other, 
    [](auto &L, auto &R) { return L * R; });
}

template <typename T>
template <typename U>
VarTT<T, std::enable_if_t<std::is_arithmetic_v<T>>> &
VarTT<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator*=(const U &ConstValue) {
  static_assert(std::is_arithmetic_v<U>, "Can only multiply VarTT by arithmetic types");
  return compoundAssignConstTT(*this, ConstValue,
    [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateMul(L, R); },
    [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateFMul(L, R); });
}

template <typename T>
template <typename U>
VarTT<T, std::enable_if_t<std::is_arithmetic_v<T>>> &
VarTT<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator/=(const VarTT<U> &Other) {
  return compoundAssignVarTT(*this, Other, 
    [](auto &L, auto &R) { return L / R; });
}

template <typename T>
template <typename U>
VarTT<T, std::enable_if_t<std::is_arithmetic_v<T>>> &
VarTT<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator/=(const U &ConstValue) {
  static_assert(std::is_arithmetic_v<U>, "Can only divide VarTT by arithmetic types");
  return compoundAssignConstTT(*this, ConstValue,
    [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateSDiv(L, R); },
    [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateFDiv(L, R); });
}

template <typename T>
template <typename U>
VarTT<T, std::enable_if_t<std::is_arithmetic_v<T>>> &
VarTT<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator%=(const VarTT<U> &Other) {
  return compoundAssignVarTT(*this, Other, 
    [](auto &L, auto &R) { return L % R; });
}

template <typename T>
template <typename U>
VarTT<T, std::enable_if_t<std::is_arithmetic_v<T>>> &
VarTT<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator%=(const U &ConstValue) {
  static_assert(std::is_arithmetic_v<U>, "Can only modulo VarTT by arithmetic types");
  return compoundAssignConstTT(*this, ConstValue,
    [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateSRem(L, R); },
    [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateFRem(L, R); });
}

// Array type operator[]
template<typename T>
VarTT<std::remove_extent_t<T>> VarTT<T, std::enable_if_t<std::is_array_v<T>>>::operator[](size_t Index) {
  auto &IRB = Fn.getIRBuilder();
  auto *ArrayTy = cast<ArrayType>(Storage->getAllocatedType());
  auto *BasePointer = Storage->getValue();
  
  // GEP into the array aggregate: [0, Index]
  auto *GEP = IRB.CreateConstInBoundsGEP2_64(ArrayTy, BasePointer, 0, Index);
  Type *ElemTy = Storage->getValueType();
  auto *BasePtrTy = cast<PointerType>(BasePointer->getType());
  unsigned AddrSpace = BasePtrTy->getAddressSpace();
  Type *ElemPtrTy = PointerType::get(ElemTy, AddrSpace);
  
  auto *PtrSlot = Fn.emitAlloca(ElemPtrTy, "elem.ptr");
  IRB.CreateStore(GEP, PtrSlot);
  
  std::unique_ptr<VarStorage> ResultStorage = std::make_unique<PointerStorage>(PtrSlot, IRB, ElemTy);
  return VarTT<std::remove_extent_t<T>>(std::move(ResultStorage), Fn);
}

template<typename T>
template <typename IdxT>
std::enable_if_t<std::is_integral_v<IdxT>, VarTT<std::remove_extent_t<T>>>
VarTT<T, std::enable_if_t<std::is_array_v<T>>>::operator[](const VarTT<IdxT> &Index) {
  auto &IRB = Fn.getIRBuilder();
  auto *ArrayTy = cast<ArrayType>(Storage->getAllocatedType());
  auto *BasePointer = Storage->getValue();
  
  Value *IdxVal = Index.Storage->loadValue();
  Value *Zero = llvm::ConstantInt::get(IdxVal->getType(), 0);
  auto *GEP = IRB.CreateInBoundsGEP(ArrayTy, BasePointer, {Zero, IdxVal});
  Type *ElemTy = Storage->getValueType();
  auto *BasePtrTy = cast<PointerType>(BasePointer->getType());
  unsigned AddrSpace = BasePtrTy->getAddressSpace();
  Type *ElemPtrTy = PointerType::get(ElemTy, AddrSpace);
  
  auto *PtrSlot = Fn.emitAlloca(ElemPtrTy, "elem.ptr");
  IRB.CreateStore(GEP, PtrSlot);
  
  std::unique_ptr<VarStorage> ResultStorage = std::make_unique<PointerStorage>(PtrSlot, IRB, ElemTy);
  return VarTT<std::remove_extent_t<T>>(std::move(ResultStorage), Fn);
}

// Pointer type operator[]
template<typename T>
VarTT<std::remove_pointer_t<T>> VarTT<T, std::enable_if_t<std::is_pointer_v<T>>>::operator[](size_t Index) {
  auto &IRB = Fn.getIRBuilder();

  auto *AllocatedType = Storage->getAllocatedType();
  auto *PointerElemTy = Storage->getValueType();
  auto *Ptr = Storage->getValue();
  auto *GEP = IRB.CreateConstInBoundsGEP1_64(PointerElemTy, Ptr, Index);
  unsigned AddrSpace = cast<PointerType>(Ptr->getType())->getAddressSpace();
  Type *ElemPtrTy = PointerType::get(PointerElemTy, AddrSpace);
  
  // Create a pointer storage to hold the LValue for
  // the Array[Index].
  auto *PtrSlot = Fn.emitAlloca(ElemPtrTy, "elem.ptr");
  IRB.CreateStore(GEP, PtrSlot);
  std::unique_ptr<VarStorage> ResultStorage = std::make_unique<PointerStorage>(PtrSlot, IRB, PointerElemTy);

  return VarTT<std::remove_pointer_t<T>>(std::move(ResultStorage), Fn);
}

template<typename T>
template <typename IdxT>
std::enable_if_t<std::is_arithmetic_v<IdxT>, VarTT<std::remove_pointer_t<T>>>
VarTT<T, std::enable_if_t<std::is_pointer_v<T>>>::operator[](const VarTT<IdxT> &Index) {
  auto &IRB = Fn.getIRBuilder();

  auto *PointeeType = Storage->getValueType();
  auto *Ptr = Storage->getValue();
  auto *IdxValue = IRB.CreateLoad(Index.Storage->getAllocatedType(), Index.Storage->getValue());
  auto *GEP = IRB.CreateInBoundsGEP(PointeeType, Ptr, IdxValue);
  unsigned AddrSpace = cast<PointerType>(Ptr->getType())->getAddressSpace();
  Type *ElemPtrTy = PointerType::get(PointeeType, AddrSpace);
  
  // Create a pointer storage to hold the LValue for
  // the Array[Index].
  auto *PtrSlot = Fn.emitAlloca(ElemPtrTy, "elem.ptr");
  IRB.CreateStore(GEP, PtrSlot);
  std::unique_ptr<VarStorage> ResultStorage = std::make_unique<PointerStorage>(PtrSlot, IRB, PointeeType);

  return VarTT<std::remove_pointer_t<T>>(std::move(ResultStorage), Fn);
}

// Pointer type operator*
template<typename T>
VarTT<std::remove_pointer_t<T>> VarTT<T, std::enable_if_t<std::is_pointer_v<T>>>::operator*() {
  return (*this)[0];
}

// Comparison operators for VarTT
template <typename T>
template <typename U>
std::enable_if_t<std::is_arithmetic_v<U>, VarTT<bool>>
VarTT<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator>(const VarTT<U> &Other) const {
  return cmpOpTT(*this, Other,
    [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateICmpSGT(L, R); },
    [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateFCmpOGT(L, R); });
}

template <typename T>
template <typename U>
std::enable_if_t<std::is_arithmetic_v<U>, VarTT<bool>>
VarTT<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator>=(const VarTT<U> &Other) const {
  return cmpOpTT(*this, Other,
    [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateICmpSGE(L, R); },
    [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateFCmpOGE(L, R); });
}

template <typename T>
template <typename U>
std::enable_if_t<std::is_arithmetic_v<U>, VarTT<bool>>
VarTT<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator<(const VarTT<U> &Other) const {
  return cmpOpTT(*this, Other,
    [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateICmpSLT(L, R); },
    [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateFCmpOLT(L, R); });
}

template <typename T>
template <typename U>
std::enable_if_t<std::is_arithmetic_v<U>, VarTT<bool>>
VarTT<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator<=(const VarTT<U> &Other) const {
  return cmpOpTT(*this, Other,
    [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateICmpSLE(L, R); },
    [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateFCmpOLE(L, R); });
}

template <typename T>
template <typename U>
std::enable_if_t<std::is_arithmetic_v<U>, VarTT<bool>>
VarTT<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator==(const VarTT<U> &Other) const {
  return cmpOpTT(*this, Other,
    [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateICmpEQ(L, R); },
    [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateFCmpOEQ(L, R); });
}

template <typename T>
template <typename U>
std::enable_if_t<std::is_arithmetic_v<U>, VarTT<bool>>
VarTT<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator!=(const VarTT<U> &Other) const {
  return cmpOpTT(*this, Other,
    [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateICmpNE(L, R); },
    [](IRBuilderBase &B, Value *L, Value *R) { return B.CreateFCmpONE(L, R); });
}

template <typename T>
template <typename U>
std::enable_if_t<std::is_arithmetic_v<U>, VarTT<bool>>
VarTT<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator>(const U &ConstValue) const {
  VarTT<U> Tmp = Fn.defVarTT<U>(ConstValue, "cmp.");
  return (*this) > Tmp;
}

template <typename T>
template <typename U>
std::enable_if_t<std::is_arithmetic_v<U>, VarTT<bool>>
VarTT<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator>=(const U &ConstValue) const {
  VarTT<U> Tmp = Fn.defVarTT<U>(ConstValue, "cmp.");
  return (*this) >= Tmp;
}

template <typename T>
template <typename U>
std::enable_if_t<std::is_arithmetic_v<U>, VarTT<bool>>
VarTT<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator<(const U &ConstValue) const {
  VarTT<U> Tmp = Fn.defVarTT<U>(ConstValue, "cmp.");
  return (*this) < Tmp;
}

template <typename T>
template <typename U>
std::enable_if_t<std::is_arithmetic_v<U>, VarTT<bool>>
VarTT<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator<=(const U &ConstValue) const {
  auto Tmp = Fn.defVarTT<U>(ConstValue, "cmp.");
  return (*this) <= Tmp;
}

template <typename T>
template <typename U>
std::enable_if_t<std::is_arithmetic_v<U>, VarTT<bool>>
VarTT<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator==(const U &ConstValue) const {
  VarTT<U> Tmp = Fn.defVarTT<U>(ConstValue, "cmp.");
  return (*this) == Tmp;
}

template <typename T>
template <typename U>
std::enable_if_t<std::is_arithmetic_v<U>, VarTT<bool>>
VarTT<T, std::enable_if_t<std::is_arithmetic_v<T>>>::operator!=(const U &ConstValue) const {
  auto Tmp = Fn.defVarTT<U>(ConstValue, "cmp.");
  return (*this) != Tmp;
}

} // namespace proteus

#endif // PROTEUS_FRONTEND_FUNC_HPP
