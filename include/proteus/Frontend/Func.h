#ifndef PROTEUS_FRONTEND_FUNC_H
#define PROTEUS_FRONTEND_FUNC_H

#include "proteus/AddressSpace.h"
#include "proteus/Error.h"
#include "proteus/Frontend/Dispatcher.h"
#include "proteus/Frontend/LLVMCodeBuilder.h"
#include "proteus/Frontend/TargetModel.h"
#include "proteus/Frontend/TypeMap.h"
#include "proteus/Frontend/TypeTraits.h"
#include "proteus/Frontend/Var.h"
#include "proteus/Frontend/VarStorage.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace llvm {
class BasicBlock;
class Function;
class Type;
class Value;
class LLVMContext;
} // namespace llvm

namespace proteus {

// NOLINTBEGIN(readability-identifier-naming)
template <typename T>
inline constexpr bool is_mutable_v =
    !std::is_const_v<std::remove_reference_t<T>>;
// NOLINTEND(readability-identifier-naming)

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

struct EmptyLambda {
  void operator()() const {}
};

enum class EmissionPolicy { Eager, Lazy };

// Returned by forLoop<Eager> so that buildLoopNest's static_assert fires with
// a clear message instead of a generic "cannot form a reference to void" error.
struct EmittedLoopTag {};

template <typename T> struct IsForLoopBuilder : std::false_type {};
template <typename T, typename BodyLambda>
struct IsForLoopBuilder<ForLoopBuilder<T, BodyLambda>> : std::true_type {};

class FuncBase {
public:
  JitModule &getJitModule() { return J; }

  llvm::LLVMContext &getContext();

  /// Get the underlying LLVMCodeBuilder for direct IR generation.
  LLVMCodeBuilder &getCodeBuilder();

  // Storage creation operations.
  std::unique_ptr<ScalarStorage> createScalarStorage(const std::string &Name,
                                                     llvm::Type *AllocaTy);
  std::unique_ptr<PointerStorage> createPointerStorage(const std::string &Name,
                                                       llvm::Type *AllocaTy,
                                                       llvm::Type *ElemTy);
  std::unique_ptr<ArrayStorage> createArrayStorage(const std::string &Name,
                                                   AddressSpace AS,
                                                   llvm::Type *ArrTy);

  // Conversion operations.
  template <typename FromT, typename ToT> llvm::Value *convert(llvm::Value *V) {
    using From = remove_cvref_t<FromT>;
    using To = remove_cvref_t<ToT>;
    static_assert(std::is_arithmetic_v<From>, "From type must be arithmetic");
    static_assert(std::is_arithmetic_v<To>, "To type must be arithmetic");

    auto &Ctx = getContext();

    if constexpr (std::is_same_v<From, To>) {
      return V;
    }

    llvm::Type *DestTy = TypeMap<To>::get(Ctx);

    if constexpr (std::is_integral_v<From> && std::is_floating_point_v<To>) {
      if constexpr (std::is_signed_v<From>) {
        return CB->createSIToFP(V, DestTy);
      }

      return CB->createUIToFP(V, DestTy);
    }

    if constexpr (std::is_floating_point_v<From> && std::is_integral_v<To>) {
      if constexpr (std::is_signed_v<To>) {
        return CB->createFPToSI(V, DestTy);
      }

      return CB->createFPToUI(V, DestTy);
    }

    if constexpr (std::is_integral_v<From> && std::is_integral_v<To>) {
      // LLVMCodeBuilder::createIntCast handles Trunc/SExt/ZExt logic internally
      return CB->createIntCast(V, DestTy, std::is_signed_v<From>);
    }

    if constexpr (std::is_floating_point_v<From> &&
                  std::is_floating_point_v<To>) {
      // LLVMCodeBuilder::createFPCast handles FPExt/FPTrunc logic internally
      return CB->createFPCast(V, DestTy);
    }

    reportFatalError("Unsupported conversion");
  }

protected:
  JitModule &J;

  std::string Name;
  LLVMCodeBuilder *CB;
  llvm::Function *LLVMFunc;

public:
  FuncBase(JitModule &J, LLVMCodeBuilder &CB, const std::string &Name,
           llvm::Type *RetTy, const std::vector<llvm::Type *> &ArgTys);
  ~FuncBase();

  TargetModelType getTargetModel() const;

  llvm::Function *getFunction();
  llvm::Value *getArg(size_t Idx);

#if PROTEUS_ENABLE_CUDA || PROTEUS_ENABLE_HIP
  // Kernel management.
  void setKernel();
  void setLaunchBoundsForKernel(int MaxThreadsPerBlock, int MinBlocksPerSM);
#endif

  template <typename T> Var<T> declVar(const std::string &Name = "var") {
    static_assert(!std::is_array_v<T>, "Expected non-array type");
    static_assert(!std::is_reference_v<T>,
                  "declVar does not support reference types");

    auto &Ctx = getContext();
    llvm::Type *AllocaTy = TypeMap<T>::get(Ctx);

    if constexpr (std::is_pointer_v<T>) {
      llvm::Type *PtrElemTy = TypeMap<T>::getPointerElemType(Ctx);
      return Var<T>{createPointerStorage(Name, AllocaTy, PtrElemTy), *this};
    } else {
      return Var<T>{createScalarStorage(Name, AllocaTy), *this};
    }
  }

  template <typename T>
  Var<T> declVar(size_t NElem, AddressSpace AS = AddressSpace::DEFAULT,
                 const std::string &Name = "array_var") {
    static_assert(std::is_array_v<T>, "Expected array type");

    auto *ArrTy = TypeMap<T>::get(getContext(), NElem);

    return Var<T>{createArrayStorage(Name, AS, ArrTy), *this};
  }

  template <typename... Ts> auto declVars() {
    return std::make_tuple(declVar<Ts>()...);
  }

  template <typename... Ts, typename... NameTs>
  auto declVars(NameTs &&...Names) {
    static_assert(sizeof...(Ts) == sizeof...(NameTs),
                  "Number of types must match number of names");
    return std::make_tuple(declVar<Ts>(std::forward<NameTs>(Names))...);
  }

  template <typename T>
  Var<T> defVar(const T &Val, const std::string &Name = "var") {
    using RawT = std::remove_const_t<T>;
    Var<RawT> V = declVar<RawT>(Name);
    V = Val;
    return Var<T>(V);
  }

  template <typename T, typename U>
  Var<T> defVar(const Var<U> &Val, const std::string &Name = "var") {
    using RawT = std::remove_const_t<T>;
    Var<RawT> Res = declVar<RawT>(Name);
    Res = Val;
    return Var<T>(Res);
  }

  template <typename U>
  Var<U> defVar(const Var<U> &Val, const std::string &Name = "var") {
    Var<U> Res = declVar<U>(Name);
    Res = Val;
    return Res;
  }

  template <
      typename T, typename NameT,
      typename = std::enable_if_t<std::is_convertible_v<NameT, std::string>>>
  auto defVar(std::pair<T, NameT> P) {
    return defVar(P.first, std::string(P.second));
  }

  template <typename... ArgT> auto defVars(ArgT &&...Args) {
    return std::make_tuple(defVar(std::forward<ArgT>(Args))...);
  }

  template <typename T>
  Var<const T> defRuntimeConst(const T &Val,
                               const std::string &Name = "run.const.var") {
    return Var<const T>(defVar<T>(Val, Name));
  }

  template <
      typename T, typename NameT,
      typename = std::enable_if_t<std::is_convertible_v<NameT, std::string>>>
  Var<const T> defRuntimeConst(std::pair<T, NameT> P) {
    return defRuntimeConst(P.first, std::string(P.second));
  }

  template <typename... ArgT> auto defRuntimeConsts(ArgT &&...Args) {
    return std::make_tuple(defRuntimeConst(std::forward<ArgT>(Args))...);
  }

  void beginFunction(const char *File = __builtin_FILE(),
                     int Line = __builtin_LINE());
  void endFunction();

  template <typename BodyLambda>
  void function(BodyLambda &&Body, const char *File = __builtin_FILE(),
                int Line = __builtin_LINE()) {
    beginFunction(File, Line);
    std::forward<BodyLambda>(Body)();
    endFunction();
  }

  void beginIf(const Var<bool> &CondVar, const char *File = __builtin_FILE(),
               int Line = __builtin_LINE());
  void endIf();

  template <typename BodyLambda>
  void ifThen(const Var<bool> &CondVar, BodyLambda &&Body,
              const char *File = __builtin_FILE(),
              int Line = __builtin_LINE()) {
    beginIf(CondVar, File, Line);
    std::forward<BodyLambda>(Body)();
    endIf();
  }

  template <typename IterT, typename InitT, typename UpperT, typename IncT>
  void beginFor(Var<IterT> &IterVar, const Var<InitT> &InitVar,
                const Var<UpperT> &UpperBound, const Var<IncT> &IncVar,
                const char *File = __builtin_FILE(),
                int Line = __builtin_LINE());
  void endFor();

  template <typename CondLambda>
  void beginWhile(CondLambda &&Cond, const char *File = __builtin_FILE(),
                  int Line = __builtin_LINE());
  void endWhile();

  template <typename CondLambda, typename BodyLambda>
  void whileLoop(CondLambda &&Cond, BodyLambda &&Body,
                 const char *File = __builtin_FILE(),
                 int Line = __builtin_LINE()) {
    beginWhile(std::forward<CondLambda>(Cond), File, Line);
    std::forward<BodyLambda>(Body)();
    endWhile();
  }

  template <typename Sig>
  std::enable_if_t<!std::is_void_v<typename FnSig<Sig>::RetT>,
                   Var<typename FnSig<Sig>::RetT>>
  call(const std::string &Name);

  template <typename Sig>
  std::enable_if_t<std::is_void_v<typename FnSig<Sig>::RetT>, void>
  call(const std::string &Name);

  template <typename Sig, typename... ArgVars>
  std::enable_if_t<!std::is_void_v<typename FnSig<Sig>::RetT>,
                   Var<typename FnSig<Sig>::RetT>>
  call(const std::string &Name, ArgVars &&...ArgsVars);

  template <typename Sig, typename... ArgVars>
  std::enable_if_t<std::is_void_v<typename FnSig<Sig>::RetT>, void>
  call(const std::string &Name, ArgVars &&...ArgsVars);

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
  std::enable_if_t<is_arithmetic_unref_v<T>, Var<T>>
  atomicAdd(const Var<T *> &Addr, const Var<T> &Val);
  template <typename T>
  std::enable_if_t<is_arithmetic_unref_v<T>, Var<T>>
  atomicSub(const Var<T *> &Addr, const Var<T> &Val);
  template <typename T>
  std::enable_if_t<is_arithmetic_unref_v<T>, Var<T>>
  atomicMax(const Var<T *> &Addr, const Var<T> &Val);
  template <typename T>
  std::enable_if_t<is_arithmetic_unref_v<T>, Var<T>>
  atomicMin(const Var<T *> &Addr, const Var<T> &Val);

  template <EmissionPolicy Policy = EmissionPolicy::Eager, typename IterT,
            typename InitT, typename UpperT, typename IncT,
            typename BodyLambda = EmptyLambda>
  auto forLoop(Var<IterT> &Iter, const Var<InitT> &Init,
               const Var<UpperT> &Upper, const Var<IncT> &Inc,
               BodyLambda &&Body = {}) {
    static_assert(is_mutable_v<IterT>, "Loop iterator must be mutable");
    if constexpr (Policy == EmissionPolicy::Eager) {
      beginFor(Iter, Init, Upper, Inc);
      std::forward<BodyLambda>(Body)();
      endFor();
      return EmittedLoopTag{};
    } else {
      LoopBoundInfo<IterT> BoundsInfo{Iter, Init, Upper, Inc};
      return ForLoopBuilder<IterT, BodyLambda>(BoundsInfo, *this,
                                               std::forward<BodyLambda>(Body));
    }
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

  const std::string &getName() const { return Name; }

  void setName(const std::string &NewName);

  // Convert the given Var's value to type U and return a new Var holding
  // the converted value. Preserves cv-qualifiers but drops references.
  template <typename U, typename T>
  std::enable_if_t<std::is_convertible_v<std::remove_reference_t<T>,
                                         std::remove_reference_t<U>>,
                   Var<std::remove_reference_t<U>>>
  convert(const Var<T> &V) {
    using ResultT = std::remove_reference_t<U>;
    Var<ResultT> Res = declVar<ResultT>("convert.");
    llvm::Value *Converted = convert<T, U>(V.loadValue());
    Res.storeValue(Converted);
    return Res;
  }
};

template <typename RetT, typename... ArgT> class Func final : public FuncBase {
private:
  Dispatcher &Dispatch;
  RetT (*CompiledFunc)(ArgT...) = nullptr;
  // Optional because Var<ArgT> is not default constructible.
  std::tuple<std::optional<Var<ArgT>>...> ArgumentsT;

private:
  template <typename T, std::size_t ArgIdx> Var<T> createArg() {
    auto Var = declVar<T>("arg." + std::to_string(ArgIdx));
    if constexpr (std::is_pointer_v<T>) {
      Var.storePointer(FuncBase::getArg(ArgIdx));
    } else {
      Var.storeValue(FuncBase::getArg(ArgIdx));
    }
    return Var;
  }

  template <std::size_t... Is> void declArgsImpl(std::index_sequence<Is...>) {
    CB->setInsertPointAtEntry();

    (std::get<Is>(ArgumentsT).emplace(createArg<ArgT, Is>()), ...);

    CB->clearInsertPoint();
  }

  template <std::size_t... Is> auto getArgsImpl(std::index_sequence<Is...>) {
    return std::tie(*std::get<Is>(ArgumentsT)...);
  }

public:
  Func(JitModule &J, LLVMCodeBuilder &CB, const std::string &Name,
       Dispatcher &Dispatch)
      : FuncBase(J, CB, Name, TypeMap<RetT>::get(CB.getContext()),
                 {TypeMap<ArgT>::get(CB.getContext())...}),
        Dispatch(Dispatch) {}

  RetT operator()(ArgT... Args);

  void declArgs() { declArgsImpl(std::index_sequence_for<ArgT...>{}); }

  auto getArgs() { return getArgsImpl(std::index_sequence_for<ArgT...>{}); }

  template <std::size_t Idx> auto &getArg() {
    return *std::get<Idx>(ArgumentsT);
  }

  auto getCompiledFunc() const { return CompiledFunc; }

  void setCompiledFunc(RetT (*CompiledFuncIn)(ArgT...)) {
    CompiledFunc = CompiledFuncIn;
  }
};

// beginFor implementation
template <typename IterT, typename InitT, typename UpperT, typename IncT>
void FuncBase::beginFor(Var<IterT> &IterVar, const Var<InitT> &Init,
                        const Var<UpperT> &UpperBound, const Var<IncT> &Inc,
                        const char *File, int Line) {
  static_assert(std::is_integral_v<std::remove_const_t<IterT>>,
                "Loop iterator must be an integral type");
  static_assert(is_mutable_v<IterT>, "Loop iterator must be mutable");

  // Update the terminator of the current basic block due to the split
  // control-flow.
  auto [CurBlock, NextBlock] = CB->splitCurrentBlock();
  CB->pushScope(File, Line, ScopeKind::FOR, NextBlock);

  llvm::BasicBlock *Header = CB->createBasicBlock("loop.header", NextBlock);
  llvm::BasicBlock *LoopCond = CB->createBasicBlock("loop.cond", NextBlock);
  llvm::BasicBlock *Body = CB->createBasicBlock("loop.body", NextBlock);
  llvm::BasicBlock *Latch = CB->createBasicBlock("loop.inc", NextBlock);
  llvm::BasicBlock *LoopExit = CB->createBasicBlock("loop.end", NextBlock);

  // Erase the old terminator and branch to the header.
  CB->eraseTerminator(CurBlock);
  CB->setInsertPoint(CurBlock);
  { CB->createBr(Header); }

  CB->setInsertPoint(Header);
  {
    IterVar = Init;
    CB->createBr(LoopCond);
  }

  CB->setInsertPoint(LoopCond);
  {
    auto CondVar = IterVar < UpperBound;
    llvm::Value *Cond = CondVar.loadValue();
    CB->createCondBr(Cond, Body, LoopExit);
  }

  CB->setInsertPoint(Body);
  CB->createBr(Latch);

  CB->setInsertPoint(Latch);
  {
    IterVar = IterVar + Inc;
    CB->createBr(LoopCond);
  }

  CB->setInsertPoint(LoopExit);
  { CB->createBr(NextBlock); }

  CB->setInsertPointBegin(Body);
}

template <typename CondLambda>
void FuncBase::beginWhile(CondLambda &&Cond, const char *File, int Line) {
  // Update the terminator of the current basic block due to the split
  // control-flow.
  auto [CurBlock, NextBlock] = CB->splitCurrentBlock();
  CB->pushScope(File, Line, ScopeKind::WHILE, NextBlock);

  llvm::BasicBlock *LoopCond = CB->createBasicBlock("while.cond", NextBlock);
  llvm::BasicBlock *Body = CB->createBasicBlock("while.body", NextBlock);
  llvm::BasicBlock *LoopExit = CB->createBasicBlock("while.end", NextBlock);

  CB->eraseTerminator(CurBlock);
  CB->setInsertPoint(CurBlock);
  { CB->createBr(LoopCond); }

  CB->setInsertPoint(LoopCond);
  {
    auto CondVar = Cond();
    llvm::Value *CondV = CondVar.loadValue();
    CB->createCondBr(CondV, Body, LoopExit);
  }

  CB->setInsertPoint(Body);
  CB->createBr(LoopCond);

  CB->setInsertPoint(LoopExit);
  { CB->createBr(NextBlock); }

  CB->setInsertPointBegin(Body);
}

template <typename Sig, typename... ArgVars>
std::enable_if_t<std::is_void_v<typename FnSig<Sig>::RetT>, void>
FuncBase::call(const std::string &Name, ArgVars &&...ArgsVars) {
  using RetT = typename FnSig<Sig>::RetT;
  using ArgT = typename FnSig<Sig>::ArgsTList;

  auto GetArgVal = [](auto &&Arg) {
    using ArgVarT = std::decay_t<decltype(Arg)>;
    if constexpr (std::is_pointer_v<typename ArgVarT::ValueType>)
      return Arg.loadPointer();
    else
      return Arg.loadValue();
  };

  auto &Ctx = getContext();
  std::vector<llvm::Type *> ArgTys = unpackArgTypes(ArgT{}, Ctx);
  getCodeBuilder().createCall(Name, TypeMap<RetT>::get(getContext()), ArgTys,
                              {GetArgVal(ArgsVars)...});
}

template <typename Sig>
std::enable_if_t<!std::is_void_v<typename FnSig<Sig>::RetT>,
                 Var<typename FnSig<Sig>::RetT>>
FuncBase::call(const std::string &Name) {
  using RetT = typename FnSig<Sig>::RetT;

  auto *Call =
      getCodeBuilder().createCall(Name, TypeMap<RetT>::get(getContext()));
  Var<RetT> Ret = declVar<RetT>("ret");
  Ret.storeValue(Call);
  return Ret;
}

template <typename Sig>
std::enable_if_t<std::is_void_v<typename FnSig<Sig>::RetT>, void>
FuncBase::call(const std::string &Name) {
  using RetT = typename FnSig<Sig>::RetT;
  getCodeBuilder().createCall(Name, TypeMap<RetT>::get(getContext()));
}

template <typename... Ts>
std::vector<llvm::Type *> unpackArgTypes(ArgTypeList<Ts...>,
                                         llvm::LLVMContext &Ctx) {
  return {TypeMap<Ts>::get(Ctx)...};
}

template <typename Sig, typename... ArgVars>
std::enable_if_t<!std::is_void_v<typename FnSig<Sig>::RetT>,
                 Var<typename FnSig<Sig>::RetT>>
FuncBase::call(const std::string &Name, ArgVars &&...ArgsVars) {
  using RetT = typename FnSig<Sig>::RetT;
  using ArgT = typename FnSig<Sig>::ArgsTList;

  auto GetArgVal = [](auto &&Arg) {
    using ArgVarT = std::decay_t<decltype(Arg)>;
    if constexpr (std::is_pointer_v<typename ArgVarT::ValueType>)
      return Arg.loadPointer();
    else
      return Arg.loadValue();
  };

  auto &Ctx = getContext();
  std::vector<llvm::Type *> ArgTys = unpackArgTypes(ArgT{}, Ctx);
  std::vector<llvm::Value *> ArgVals = {GetArgVal(ArgsVars)...};

  auto *Call = getCodeBuilder().createCall(Name, TypeMap<RetT>::get(Ctx),
                                           ArgTys, ArgVals);

  Var<RetT> Ret = declVar<RetT>("ret");
  Ret.storeValue(Call);
  return Ret;
}

// Var implementations (defined here after FuncBase is
// complete) so we have it available.

// Helper function for binary operations on Var types
template <typename T, typename U, typename IntOp, typename FPOp>
Var<std::common_type_t<remove_cvref_t<T>, remove_cvref_t<U>>>
binOp(const Var<T> &L, const Var<U> &R, IntOp IOp, FPOp FOp) {
  using CommonT = std::common_type_t<remove_cvref_t<T>, remove_cvref_t<U>>;

  FuncBase &Fn = L.Fn;
  if (&Fn != &R.Fn)
    reportFatalError("Variables should belong to the same function");

  llvm::Value *LHS = Fn.convert<T, CommonT>(L.loadValue());
  llvm::Value *RHS = Fn.convert<U, CommonT>(R.loadValue());

  llvm::Value *Result = nullptr;
  if constexpr (std::is_integral_v<CommonT>) {
    Result = IOp(Fn.getCodeBuilder(), LHS, RHS);
  } else {
    Result = FOp(Fn.getCodeBuilder(), LHS, RHS);
  }

  auto ResultVar = Fn.declVar<CommonT>("res.");
  ResultVar.storeValue(Result);

  return ResultVar;
}

// Helper function for compound assignment with a constant
template <typename T, typename U, typename IntOp, typename FPOp>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &
compoundAssignConst(Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &LHS,
                    const U &ConstValue, IntOp IOp, FPOp FOp) {
  static_assert(std::is_convertible_v<remove_cvref_t<U>, remove_cvref_t<T>>,
                "U must be convertible to T");

  auto &Ctx = LHS.Fn.getContext();
  llvm::Type *RHSType = TypeMap<remove_cvref_t<U>>::get(Ctx);

  llvm::Value *RHS = nullptr;
  if constexpr (std::is_integral_v<remove_cvref_t<U>>) {
    RHS = LHS.Fn.getCodeBuilder().getConstantInt(RHSType, ConstValue);
  } else {
    RHS = LHS.Fn.getCodeBuilder().getConstantFP(RHSType, ConstValue);
  }

  llvm::Value *LHSVal = LHS.loadValue();

  RHS = LHS.Fn.template convert<U, T>(RHS);
  llvm::Value *Result = nullptr;

  if constexpr (std::is_integral_v<remove_cvref_t<T>>) {
    Result = IOp(LHS.Fn.getCodeBuilder(), LHSVal, RHS);
  } else {
    static_assert(std::is_floating_point_v<remove_cvref_t<T>>,
                  "Unsupported type");
    Result = FOp(LHS.Fn.getCodeBuilder(), LHSVal, RHS);
  }

  LHS.storeValue(Result);
  return LHS;
}

// Helper function for comparison operations on Var types
template <typename T, typename U, typename IntOp, typename FPOp>
Var<bool> cmpOp(const Var<T> &L, const Var<U> &R, IntOp IOp, FPOp FOp) {
  FuncBase &Fn = L.Fn;
  if (&Fn != &R.Fn)
    reportFatalError("Variables should belong to the same function");

  llvm::Value *LHS = L.loadValue();
  llvm::Value *RHS = Fn.convert<U, T>(R.loadValue());

  llvm::Value *Result = nullptr;
  if constexpr (std::is_integral_v<remove_cvref_t<T>>) {
    Result = IOp(Fn.getCodeBuilder(), LHS, RHS);
  } else {
    static_assert(std::is_floating_point_v<remove_cvref_t<T>>,
                  "Unsupported type");
    Result = FOp(Fn.getCodeBuilder(), LHS, RHS);
  }

  auto ResultVar = Fn.declVar<bool>("res.");
  ResultVar.storeValue(Result);

  return ResultVar;
}

template <typename T>
template <typename U, typename>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::Var(const Var<U> &V)
    : VarStorageOwner<VarStorage>(V.Fn) {
  // Allocate storage for the target type T.
  llvm::Type *TargetTy = TypeMap<remove_cvref_t<T>>::get(Fn.getContext());
  Storage = Fn.createScalarStorage("conv.var", TargetTy);

  auto *Converted = Fn.convert<U, T>(V.loadValue());
  storeValue(Converted);
}

template <typename T>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator=(const Var &V) {
  static_assert(is_mutable_v<T>, "Cannot assign to Var<const T>");
  storeValue(V.loadValue());
  return *this;
}

template <typename T>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator=(Var &&V) {
  static_assert(is_mutable_v<T>, "Cannot assign to Var<const T>");
  if (this->Storage == nullptr) {
    // If we don't have storage, clone it from the source.
    Storage = V.Storage->clone();
  } else {
    // If we have storage, copy the value.
    storeValue(V.loadValue());
  }
  return *this;
}

template <typename T>
Var<std::add_pointer_t<T>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::getAddress() {
  if constexpr (std::is_reference_v<T>) {
    // For references, load the pointer to create a T*.
    auto *PtrStorage = static_cast<PointerStorage *>(Storage.get());
    llvm::Value *PtrVal = PtrStorage->loadPointer();
    llvm::Type *ElemTy = PtrStorage->getValueType();
    unsigned AddrSpace = Fn.getCodeBuilder().getAddressSpaceFromValue(PtrVal);
    llvm::Type *PtrTy = Fn.getCodeBuilder().getPointerType(ElemTy, AddrSpace);
    PtrVal = Fn.getCodeBuilder().createBitCast(PtrVal, PtrTy);

    std::unique_ptr<PointerStorage> ResultStorage =
        Fn.createPointerStorage("addr.ref.tmp", PtrTy, ElemTy);
    Fn.getCodeBuilder().createStore(PtrVal, ResultStorage->getSlot());

    return Var<std::add_pointer_t<T>>(std::move(ResultStorage), Fn);
  }

  llvm::Value *Slot = getSlot();
  llvm::Type *ElemTy = getAllocatedType();

  unsigned AddrSpace = Fn.getCodeBuilder().getAddressSpace(getSlotType());
  llvm::Type *PtrTy = Fn.getCodeBuilder().getPointerType(ElemTy, AddrSpace);
  llvm::Value *PtrVal = Slot;
  PtrVal = Fn.getCodeBuilder().createBitCast(Slot, PtrTy);

  std::unique_ptr<PointerStorage> ResultStorage =
      Fn.createPointerStorage("addr.tmp", PtrTy, ElemTy);
  Fn.getCodeBuilder().createStore(PtrVal, ResultStorage->getSlot());

  return Var<std::add_pointer_t<T>>(std::move(ResultStorage), Fn);
}

template <typename T>
template <typename U>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator=(
    const Var<U> &V) {
  static_assert(is_mutable_v<T>, "Cannot assign to Var<const T>");
  auto *Converted = Fn.convert<U, T>(V.loadValue());
  storeValue(Converted);
  return *this;
}

template <typename T>
template <typename U>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator=(
    const U &ConstValue) {
  static_assert(is_mutable_v<T>, "Cannot assign to Var<const T>");
  static_assert(std::is_arithmetic_v<U>,
                "Can only assign arithmetic types to Var");

  llvm::Type *LHSType = getValueType();

  if (Fn.getCodeBuilder().isIntegerTy(LHSType)) {
    storeValue(Fn.getCodeBuilder().getConstantInt(LHSType, ConstValue));
  } else if (Fn.getCodeBuilder().isFloatingPointTy(LHSType)) {
    storeValue(Fn.getCodeBuilder().getConstantFP(LHSType, ConstValue));
  } else {
    reportFatalError("Unsupported type");
  }

  return *this;
}

template <typename T>
template <typename U>
Var<std::common_type_t<T, U>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator+(
    const Var<U> &Other) const {
  return binOp(
      *this, Other,
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
        return CB.createAdd(L, R);
      },
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
        return CB.createFAdd(L, R);
      });
}

template <typename T>
template <typename U>
Var<std::common_type_t<T, U>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator-(
    const Var<U> &Other) const {
  return binOp(
      *this, Other,
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
        return CB.createSub(L, R);
      },
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
        return CB.createFSub(L, R);
      });
}

template <typename T>
template <typename U>
Var<std::common_type_t<T, U>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator*(
    const Var<U> &Other) const {
  return binOp(
      *this, Other,
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
        return CB.createMul(L, R);
      },
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
        return CB.createFMul(L, R);
      });
}

template <typename T>
template <typename U>
Var<std::common_type_t<T, U>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator/(
    const Var<U> &Other) const {
  return binOp(
      *this, Other,
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
        return CB.createSDiv(L, R);
      },
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
        return CB.createFDiv(L, R);
      });
}

template <typename T>
template <typename U>
Var<std::common_type_t<T, U>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator%(
    const Var<U> &Other) const {
  return binOp(
      *this, Other,
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
        return CB.createSRem(L, R);
      },
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
        return CB.createFRem(L, R);
      });
}

// Arithmetic operators with ConstValue
template <typename T>
template <typename U>
std::enable_if_t<std::is_arithmetic_v<U>, Var<std::common_type_t<T, U>>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator+(
    const U &ConstValue) const {
  static_assert(std::is_arithmetic_v<U>,
                "Can only add arithmetic types to Var");
  Var<U> Tmp = Fn.defVar<U>(ConstValue, "tmp.");
  return (*this) + Tmp;
}

template <typename T>
template <typename U>
std::enable_if_t<std::is_arithmetic_v<U>, Var<std::common_type_t<T, U>>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator-(
    const U &ConstValue) const {
  static_assert(std::is_arithmetic_v<U>,
                "Can only subtract arithmetic types from Var");
  Var<U> Tmp = Fn.defVar<U>(ConstValue, "tmp.");
  return (*this) - Tmp;
}

template <typename T>
template <typename U>
std::enable_if_t<std::is_arithmetic_v<U>, Var<std::common_type_t<T, U>>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator*(
    const U &ConstValue) const {
  static_assert(std::is_arithmetic_v<U>,
                "Can only multiply Var by arithmetic types");
  Var<U> Tmp = Fn.defVar<U>(ConstValue, "tmp.");
  return (*this) * Tmp;
}

template <typename T>
template <typename U>
std::enable_if_t<std::is_arithmetic_v<U>, Var<std::common_type_t<T, U>>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator/(
    const U &ConstValue) const {
  static_assert(std::is_arithmetic_v<U>,
                "Can only divide Var by arithmetic types");
  Var<U> Tmp = Fn.defVar<U>(ConstValue, "tmp.");
  return (*this) / Tmp;
}

template <typename T>
template <typename U>
std::enable_if_t<std::is_arithmetic_v<U>, Var<std::common_type_t<T, U>>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator%(
    const U &ConstValue) const {
  static_assert(std::is_arithmetic_v<U>,
                "Can only modulo Var by arithmetic types");
  Var<U> Tmp = Fn.defVar<U>(ConstValue, "tmp.");
  return (*this) % Tmp;
}

// Compound assignment operators for Var
template <typename T>
template <typename U>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator+=(
    const Var<U> &Other) {
  static_assert(is_mutable_v<T>, "Cannot use += on Var<const T>");
  auto Result = (*this) + Other;
  *this = Result;
  return *this;
}

template <typename T>
template <typename U>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator+=(
    const U &ConstValue) {
  static_assert(is_mutable_v<T>, "Cannot use += on Var<const T>");
  static_assert(std::is_arithmetic_v<U>,
                "Can only add arithmetic types to Var");
  return compoundAssignConst(
      *this, ConstValue,
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
        return CB.createAdd(L, R);
      },
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
        return CB.createFAdd(L, R);
      });
}

template <typename T>
template <typename U>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator-=(
    const Var<U> &Other) {
  static_assert(is_mutable_v<T>, "Cannot use -= on Var<const T>");
  auto Result = (*this) - Other;
  *this = Result;
  return *this;
}

template <typename T>
template <typename U>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator-=(
    const U &ConstValue) {
  static_assert(is_mutable_v<T>, "Cannot use -= on Var<const T>");
  static_assert(std::is_arithmetic_v<U>,
                "Can only subtract arithmetic types from Var");
  return compoundAssignConst(
      *this, ConstValue,
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
        return CB.createSub(L, R);
      },
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
        return CB.createFSub(L, R);
      });
}

template <typename T>
template <typename U>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator*=(
    const Var<U> &Other) {
  static_assert(is_mutable_v<T>, "Cannot use *= on Var<const T>");
  auto Result = (*this) * Other;
  *this = Result;
  return *this;
}

template <typename T>
template <typename U>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator*=(
    const U &ConstValue) {
  static_assert(is_mutable_v<T>, "Cannot use *= on Var<const T>");
  static_assert(std::is_arithmetic_v<U>,
                "Can only multiply Var by arithmetic types");
  return compoundAssignConst(
      *this, ConstValue,
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
        return CB.createMul(L, R);
      },
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
        return CB.createFMul(L, R);
      });
}

template <typename T>
template <typename U>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator/=(
    const Var<U> &Other) {
  static_assert(is_mutable_v<T>, "Cannot use /= on Var<const T>");
  auto Result = (*this) / Other;
  *this = Result;
  return *this;
}

template <typename T>
template <typename U>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator/=(
    const U &ConstValue) {
  static_assert(is_mutable_v<T>, "Cannot use /= on Var<const T>");
  static_assert(std::is_arithmetic_v<U>,
                "Can only divide Var by arithmetic types");
  return compoundAssignConst(
      *this, ConstValue,
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
        return CB.createSDiv(L, R);
      },
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
        return CB.createFDiv(L, R);
      });
}

template <typename T>
template <typename U>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator%=(
    const Var<U> &Other) {
  static_assert(is_mutable_v<T>, "Cannot use %= on Var<const T>");
  auto Result = (*this) % Other;
  *this = Result;
  return *this;
}

template <typename T>
template <typename U>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator%=(
    const U &ConstValue) {
  static_assert(is_mutable_v<T>, "Cannot use %= on Var<const T>");
  static_assert(std::is_arithmetic_v<U>,
                "Can only modulo Var by arithmetic types");
  return compoundAssignConst(
      *this, ConstValue,
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
        return CB.createSRem(L, R);
      },
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
        return CB.createFRem(L, R);
      });
}

template <typename T>
Var<remove_cvref_t<T>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator-() const {
  auto MinusOne = Fn.defVar<remove_cvref_t<T>>(
      static_cast<remove_cvref_t<T>>(-1), "minus_one.");
  return MinusOne * (*this);
}

template <typename T>
Var<bool>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator!() const {
  llvm::Value *V = loadValue();
  llvm::Value *ResV = nullptr;
  if constexpr (std::is_same_v<remove_cvref_t<T>, bool>) {
    ResV = Fn.getCodeBuilder().createNot(V);
  } else if constexpr (std::is_integral_v<remove_cvref_t<T>>) {
    llvm::Value *Zero = Fn.getCodeBuilder().getConstantInt(getValueType(), 0);
    ResV = Fn.getCodeBuilder().createICmpEQ(V, Zero);
  } else {
    llvm::Value *Zero = Fn.getCodeBuilder().getConstantFP(getValueType(), 0.0);
    ResV = Fn.getCodeBuilder().createFCmpOEQ(V, Zero);
  }
  auto Ret = Fn.declVar<bool>("not.");
  Ret.storeValue(ResV);
  return Ret;
}

template <typename T>
Var<std::add_lvalue_reference_t<std::remove_extent_t<T>>>
Var<T, std::enable_if_t<std::is_array_v<T>>>::operator[](size_t Index) {
  auto *ArrayTy = getAllocatedType();
  auto *BasePointer = getSlot();

  // GEP into the array aggregate: [0, Index]
  auto *GEP = Fn.getCodeBuilder().createConstInBoundsGEP2_64(
      ArrayTy, BasePointer, 0, Index);
  llvm::Type *ElemTy = getValueType();
  unsigned AddrSpace = Fn.getCodeBuilder().getAddressSpace(getSlotType());
  llvm::Type *ElemPtrTy = Fn.getCodeBuilder().getPointerType(ElemTy, AddrSpace);

  std::unique_ptr<PointerStorage> ResultStorage =
      Fn.createPointerStorage("elem.ptr", ElemPtrTy, ElemTy);
  Fn.getCodeBuilder().createStore(GEP, ResultStorage->getSlot());
  return Var<std::add_lvalue_reference_t<std::remove_extent_t<T>>>(
      std::move(ResultStorage), Fn);
}

template <typename T>
template <typename IdxT>
std::enable_if_t<std::is_integral_v<IdxT>,
                 Var<std::add_lvalue_reference_t<std::remove_extent_t<T>>>>
Var<T, std::enable_if_t<std::is_array_v<T>>>::operator[](
    const Var<IdxT> &Index) {
  auto *ArrayTy = getAllocatedType();
  auto *BasePointer = getSlot();

  llvm::Value *IdxVal = Index.loadValue();
  llvm::Value *Zero =
      Fn.getCodeBuilder().getConstantInt(Index.getValueType(), 0);
  auto *GEP = Fn.getCodeBuilder().createInBoundsGEP(ArrayTy, BasePointer,
                                                    {Zero, IdxVal});
  llvm::Type *ElemTy = getValueType();
  unsigned AddrSpace = Fn.getCodeBuilder().getAddressSpace(getSlotType());
  llvm::Type *ElemPtrTy = Fn.getCodeBuilder().getPointerType(ElemTy, AddrSpace);

  std::unique_ptr<VarStorage> ResultStorage =
      Fn.createPointerStorage("elem.ptr", ElemPtrTy, ElemTy);
  Fn.getCodeBuilder().createStore(GEP, ResultStorage->getSlot());

  return Var<std::add_lvalue_reference_t<std::remove_extent_t<T>>>(
      std::move(ResultStorage), Fn);
}

template <typename T>
Var<std::add_lvalue_reference_t<
    std::remove_pointer_t<std::remove_reference_t<T>>>>
Var<T, std::enable_if_t<is_pointer_unref_v<T>>>::operator[](size_t Index) {
  auto *PointerElemTy =
      TypeMap<std::remove_pointer_t<std::remove_reference_t<T>>>::get(
          Fn.getContext());
  auto *Ptr = loadPointer();
  auto *GEP =
      Fn.getCodeBuilder().createConstInBoundsGEP1_64(PointerElemTy, Ptr, Index);
  unsigned AddrSpace = Fn.getCodeBuilder().getAddressSpace(getAllocatedType());
  llvm::Type *ElemPtrTy =
      Fn.getCodeBuilder().getPointerType(PointerElemTy, AddrSpace);

  // Create a pointer storage to hold the LValue for the Array[Index].
  std::unique_ptr<PointerStorage> ResultStorage =
      Fn.createPointerStorage("elem.ptr", ElemPtrTy, PointerElemTy);
  Fn.getCodeBuilder().createStore(GEP, ResultStorage->getSlot());

  return Var<std::add_lvalue_reference_t<
      std::remove_pointer_t<std::remove_reference_t<T>>>>(
      std::move(ResultStorage), Fn);
}

template <typename T>
template <typename IdxT>
std::enable_if_t<std::is_arithmetic_v<IdxT>,
                 Var<std::add_lvalue_reference_t<
                     std::remove_pointer_t<std::remove_reference_t<T>>>>>
Var<T, std::enable_if_t<is_pointer_unref_v<T>>>::operator[](
    const Var<IdxT> &Index) {

  auto *PointeeType =
      TypeMap<std::remove_pointer_t<std::remove_reference_t<T>>>::get(
          Fn.getContext());
  auto *Ptr = loadPointer();
  auto *IdxValue = Index.loadValue();
  auto *GEP =
      Fn.getCodeBuilder().createInBoundsGEP(PointeeType, Ptr, {IdxValue});
  unsigned AddrSpace = Fn.getCodeBuilder().getAddressSpace(getAllocatedType());
  llvm::Type *ElemPtrTy =
      Fn.getCodeBuilder().getPointerType(PointeeType, AddrSpace);

  // Create a pointer storage to hold the LValue for the Array[Index].
  std::unique_ptr<PointerStorage> ResultStorage =
      Fn.createPointerStorage("elem.ptr", ElemPtrTy, PointeeType);
  Fn.getCodeBuilder().createStore(GEP, ResultStorage->getSlot());

  return Var<std::add_lvalue_reference_t<
      std::remove_pointer_t<std::remove_reference_t<T>>>>(
      std::move(ResultStorage), Fn);
}

// Pointer type operator*
template <typename T>
Var<std::add_lvalue_reference_t<
    std::remove_pointer_t<std::remove_reference_t<T>>>>
Var<T, std::enable_if_t<is_pointer_unref_v<T>>>::operator*() {
  return (*this)[0];
}

template <typename T>
Var<std::add_pointer_t<T>>
Var<T, std::enable_if_t<is_pointer_unref_v<T>>>::getAddress() {
  llvm::Value *PtrVal = loadPointer();
  llvm::Type *ElemTy = getValueType();

  unsigned AddrSpace = Fn.getCodeBuilder().getAddressSpace(getAllocatedType());
  llvm::Type *PointeePtrTy =
      Fn.getCodeBuilder().getPointerType(ElemTy, AddrSpace);
  llvm::Type *TargetPtrTy =
      Fn.getCodeBuilder().getPointerTypeUnqual(PointeePtrTy);

  PtrVal = Fn.getCodeBuilder().createBitCast(PtrVal, PointeePtrTy);

  std::unique_ptr<PointerStorage> ResultStorage =
      Fn.createPointerStorage("addr.ptr.tmp", TargetPtrTy, PointeePtrTy);
  Fn.getCodeBuilder().createStore(PtrVal, ResultStorage->getSlot());

  return Var<std::add_pointer_t<T>>(std::move(ResultStorage), Fn);
}

template <typename T>
template <typename OffsetT>
std::enable_if_t<std::is_arithmetic_v<OffsetT>,
                 Var<T, std::enable_if_t<is_pointer_unref_v<T>>>>
Var<T, std::enable_if_t<is_pointer_unref_v<T>>>::operator+(
    const Var<OffsetT> &Offset) const {
  auto *OffsetVal = Offset.loadValue();
  auto *IdxVal = Fn.convert<OffsetT, int64_t>(OffsetVal);

  auto *BasePtr = loadPointer();
  auto *ElemTy = getValueType();

  auto *GEP =
      Fn.getCodeBuilder().createInBoundsGEP(ElemTy, BasePtr, IdxVal, "ptr.add");

  unsigned AddrSpace = Fn.getCodeBuilder().getAddressSpace(getAllocatedType());
  auto *ElemPtrTy = Fn.getCodeBuilder().getPointerType(ElemTy, AddrSpace);

  std::unique_ptr<PointerStorage> ResultStorage =
      Fn.createPointerStorage("ptr.add.tmp", ElemPtrTy, ElemTy);
  ;
  Fn.getCodeBuilder().createStore(GEP, ResultStorage->getSlot());

  return Var<T, std::enable_if_t<is_pointer_unref_v<T>>>(
      std::move(ResultStorage), Fn);
}

template <typename T>
template <typename OffsetT>
std::enable_if_t<std::is_arithmetic_v<OffsetT>,
                 Var<T, std::enable_if_t<is_pointer_unref_v<T>>>>
Var<T, std::enable_if_t<is_pointer_unref_v<T>>>::operator+(
    OffsetT Offset) const {
  auto *IntTy = Fn.getCodeBuilder().getInt64Ty();
  llvm::Value *IdxVal = Fn.getCodeBuilder().getConstantInt(IntTy, Offset);

  auto *BasePtr = loadPointer();
  auto *ElemTy = getValueType();

  auto *GEP = Fn.getCodeBuilder().createInBoundsGEP(ElemTy, BasePtr, {IdxVal},
                                                    "ptr.add");

  unsigned AddrSpace = Fn.getCodeBuilder().getAddressSpace(getAllocatedType());
  auto *ElemPtrTy = Fn.getCodeBuilder().getPointerType(ElemTy, AddrSpace);

  std::unique_ptr<PointerStorage> ResultStorage =
      Fn.createPointerStorage("ptr.add.tmp", ElemPtrTy, ElemTy);
  Fn.getCodeBuilder().createStore(GEP, ResultStorage->getSlot());

  return Var<T, std::enable_if_t<is_pointer_unref_v<T>>>(
      std::move(ResultStorage), Fn);
}

// Comparison operators for Var
template <typename T>
template <typename U>
std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator>(
    const Var<U> &Other) const {
  return cmpOp(
      *this, Other,
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
        return CB.createICmpSGT(L, R);
      },
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
        return CB.createFCmpOGT(L, R);
      });
}

template <typename T>
template <typename U>
std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator>=(
    const Var<U> &Other) const {
  return cmpOp(
      *this, Other,
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
        return CB.createICmpSGE(L, R);
      },
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
        return CB.createFCmpOGE(L, R);
      });
}

template <typename T>
template <typename U>
std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator<(
    const Var<U> &Other) const {
  return cmpOp(
      *this, Other,
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
        return CB.createICmpSLT(L, R);
      },
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
        return CB.createFCmpOLT(L, R);
      });
}

template <typename T>
template <typename U>
std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator<=(
    const Var<U> &Other) const {
  return cmpOp(
      *this, Other,
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
        return CB.createICmpSLE(L, R);
      },
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
        return CB.createFCmpOLE(L, R);
      });
}

template <typename T>
template <typename U>
std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator==(
    const Var<U> &Other) const {
  return cmpOp(
      *this, Other,
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
        return CB.createICmpEQ(L, R);
      },
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
        return CB.createFCmpOEQ(L, R);
      });
}

template <typename T>
template <typename U>
std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator!=(
    const Var<U> &Other) const {
  return cmpOp(
      *this, Other,
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
        return CB.createICmpNE(L, R);
      },
      [](LLVMCodeBuilder &CB, llvm::Value *L, llvm::Value *R) {
        return CB.createFCmpONE(L, R);
      });
}

template <typename T>
template <typename U>
std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator>(
    const U &ConstValue) const {
  Var<U> Tmp = Fn.defVar<U>(ConstValue, "cmp.");
  return (*this) > Tmp;
}

template <typename T>
template <typename U>
std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator>=(
    const U &ConstValue) const {
  Var<U> Tmp = Fn.defVar<U>(ConstValue, "cmp.");
  return (*this) >= Tmp;
}

template <typename T>
template <typename U>
std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator<(
    const U &ConstValue) const {
  Var<U> Tmp = Fn.defVar<U>(ConstValue, "cmp.");
  return (*this) < Tmp;
}

template <typename T>
template <typename U>
std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator<=(
    const U &ConstValue) const {
  auto Tmp = Fn.defVar<U>(ConstValue, "cmp.");
  return (*this) <= Tmp;
}

template <typename T>
template <typename U>
std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator==(
    const U &ConstValue) const {
  Var<U> Tmp = Fn.defVar<U>(ConstValue, "cmp.");
  return (*this) == Tmp;
}

template <typename T>
template <typename U>
std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator!=(
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

template <typename T>
std::enable_if_t<is_arithmetic_unref_v<T>, Var<T>>
FuncBase::atomicAdd(const Var<T *> &Addr, const Var<T> &Val) {
  static_assert(std::is_arithmetic_v<T>, "atomicAdd requires arithmetic type");

  llvm::Value *Result =
      CB->createAtomicAdd(Addr.loadPointer(), Val.loadValue());
  auto Ret = declVar<T>("atomic.add.res.");
  Ret.storeValue(Result);
  return Ret;
}

template <typename T>
std::enable_if_t<is_arithmetic_unref_v<T>, Var<T>>
FuncBase::atomicSub(const Var<T *> &Addr, const Var<T> &Val) {
  static_assert(std::is_arithmetic_v<T>, "atomicSub requires arithmetic type");

  llvm::Value *Result =
      CB->createAtomicSub(Addr.loadPointer(), Val.loadValue());
  auto Ret = declVar<T>("atomic.sub.res.");
  Ret.storeValue(Result);
  return Ret;
}

template <typename T>
std::enable_if_t<is_arithmetic_unref_v<T>, Var<T>>
FuncBase::atomicMax(const Var<T *> &Addr, const Var<T> &Val) {
  static_assert(std::is_arithmetic_v<T>, "atomicMax requires arithmetic type");

  llvm::Value *Result =
      CB->createAtomicMax(Addr.loadPointer(), Val.loadValue());
  auto Ret = declVar<T>("atomic.max.res.");
  Ret.storeValue(Result);
  return Ret;
}

template <typename T>
std::enable_if_t<is_arithmetic_unref_v<T>, Var<T>>
FuncBase::atomicMin(const Var<T *> &Addr, const Var<T> &Val) {
  static_assert(std::is_arithmetic_v<T>, "atomicMin requires arithmetic type");

  llvm::Value *Result =
      CB->createAtomicMin(Addr.loadPointer(), Val.loadValue());
  auto Ret = declVar<T>("atomic.min.res.");
  Ret.storeValue(Result);
  return Ret;
}

inline void FuncBase::ret() { CB->createRetVoid(); }

template <typename T> void FuncBase::ret(const Var<T> &RetVal) {
  llvm::Value *RetValue = RetVal.loadValue();
  CB->createRet(RetValue);
}

// Helper struct to convert Var operands to a target type T.
// Used by emitIntrinsic to convert all operands to the intrinsic's result
// type. C++17 doesn't support template parameters on lambdas, so we use a
// struct.
template <typename T> struct IntrinsicOperandConverter {
  FuncBase &Fn;

  template <typename U> llvm::Value *operator()(const Var<U> &Operand) const {
    return Fn.convert<U, T>(Operand.loadValue());
  }
};

// Helper for emitting intrinsics with Var
template <typename T, typename... Operands>
static Var<T> emitIntrinsic(const std::string &IntrinsicName,
                            llvm::Type *ResultType, const Operands &...Ops) {
  static_assert(sizeof...(Ops) > 0, "Intrinsic requires at least one operand");

  auto &Fn = std::get<0>(std::tie(Ops...)).Fn;
  auto CheckFn = [&Fn](const auto &Operand) {
    if (&Operand.Fn != &Fn)
      reportFatalError("Variables should belong to the same function");
  };
  (CheckFn(Ops), ...);

  IntrinsicOperandConverter<T> ConvertOperand{Fn};

  // All operands are converted to the result type.
  std::vector<llvm::Type *> ArgTys(sizeof...(Ops), ResultType);
  llvm::Value *Call = Fn.getCodeBuilder().createCall(
      IntrinsicName, ResultType, ArgTys, {ConvertOperand(Ops)...});

  auto ResultVar = Fn.template declVar<T>("res.");
  ResultVar.storeValue(Call);
  return ResultVar;
}

// Math intrinsics for Var
template <typename T> Var<float> powf(const Var<float> &L, const Var<T> &R) {
  static_assert(std::is_convertible_v<T, float>,
                "powf requires floating-point type");

  auto *ResultType = R.Fn.getCodeBuilder().getFloatTy();
  auto RFloat = R.Fn.template convert<float>(R);
  std::string IntrinsicName = "llvm.pow.f32";
#if PROTEUS_ENABLE_CUDA
  if (L.Fn.getTargetModel() == TargetModelType::CUDA)
    IntrinsicName = "__nv_powf";
#endif

  return emitIntrinsic<float>(IntrinsicName, ResultType, L, RFloat);
}

template <typename T> Var<float> sqrtf(const Var<T> &R) {
  static_assert(std::is_convertible_v<T, float>,
                "sqrtf requires floating-point type");

  auto *ResultType = R.Fn.getCodeBuilder().getFloatTy();
  auto RFloat = R.Fn.template convert<float>(R);
  std::string IntrinsicName = "llvm.sqrt.f32";
#if PROTEUS_ENABLE_CUDA
  if (R.Fn.getTargetModel() == TargetModelType::CUDA)
    IntrinsicName = "__nv_sqrtf";
#endif

  return emitIntrinsic<float>(IntrinsicName, ResultType, RFloat);
}

template <typename T> Var<float> expf(const Var<T> &R) {
  static_assert(std::is_convertible_v<T, float>,
                "expf requires floating-point type");

  auto *ResultType = R.Fn.getCodeBuilder().getFloatTy();
  auto RFloat = R.Fn.template convert<float>(R);
  std::string IntrinsicName = "llvm.exp.f32";
#if PROTEUS_ENABLE_CUDA
  if (R.Fn.getTargetModel() == TargetModelType::CUDA)
    IntrinsicName = "__nv_expf";
#endif

  return emitIntrinsic<float>(IntrinsicName, ResultType, RFloat);
}

template <typename T> Var<float> sinf(const Var<T> &R) {
  static_assert(std::is_convertible_v<T, float>,
                "sinf requires floating-point type");

  auto *ResultType = R.Fn.getCodeBuilder().getFloatTy();
  auto RFloat = R.Fn.template convert<float>(R);
  std::string IntrinsicName = "llvm.sin.f32";
#if PROTEUS_ENABLE_CUDA
  if (R.Fn.getTargetModel() == TargetModelType::CUDA)
    IntrinsicName = "__nv_sinf";
#endif

  return emitIntrinsic<float>(IntrinsicName, ResultType, RFloat);
}

template <typename T> Var<float> cosf(const Var<T> &R) {
  static_assert(std::is_convertible_v<T, float>,
                "cosf requires floating-point type");

  auto *ResultType = R.Fn.getCodeBuilder().getFloatTy();
  auto RFloat = R.Fn.template convert<float>(R);
  std::string IntrinsicName = "llvm.cos.f32";
#if PROTEUS_ENABLE_CUDA
  if (R.Fn.getTargetModel() == TargetModelType::CUDA)
    IntrinsicName = "__nv_cosf";
#endif

  return emitIntrinsic<float>(IntrinsicName, ResultType, RFloat);
}

template <typename T> Var<float> fabs(const Var<T> &R) {
  static_assert(std::is_convertible_v<T, float>,
                "fabs requires floating-point type");

  auto *ResultType = R.Fn.getCodeBuilder().getFloatTy();
  auto RFloat = R.Fn.template convert<float>(R);
  std::string IntrinsicName = "llvm.fabs.f32";
#if PROTEUS_ENABLE_CUDA
  if (R.Fn.getTargetModel() == TargetModelType::CUDA)
    IntrinsicName = "__nv_fabsf";
#endif

  return emitIntrinsic<float>(IntrinsicName, ResultType, RFloat);
}

template <typename T> Var<float> truncf(const Var<T> &R) {
  static_assert(std::is_convertible_v<T, float>,
                "truncf requires floating-point type");

  auto *ResultType = R.Fn.getCodeBuilder().getFloatTy();
  auto RFloat = R.Fn.template convert<float>(R);
  std::string IntrinsicName = "llvm.trunc.f32";
#if PROTEUS_ENABLE_CUDA
  if (R.Fn.getTargetModel() == TargetModelType::CUDA)
    IntrinsicName = "__nv_truncf";
#endif

  return emitIntrinsic<float>(IntrinsicName, ResultType, RFloat);
}

template <typename T> Var<float> logf(const Var<T> &R) {
  static_assert(std::is_convertible_v<T, float>,
                "logf requires floating-point type");

  auto *ResultType = R.Fn.getCodeBuilder().getFloatTy();
  auto RFloat = R.Fn.template convert<float>(R);
  std::string IntrinsicName = "llvm.log.f32";
#if PROTEUS_ENABLE_CUDA
  if (R.Fn.getTargetModel() == TargetModelType::CUDA)
    IntrinsicName = "__nv_logf";
#endif

  return emitIntrinsic<float>(IntrinsicName, ResultType, RFloat);
}

template <typename T> Var<float> absf(const Var<T> &R) {
  static_assert(std::is_convertible_v<T, float>,
                "absf requires floating-point type");

  auto *ResultType = R.Fn.getCodeBuilder().getFloatTy();
  auto RFloat = R.Fn.template convert<float>(R);
  std::string IntrinsicName = "llvm.fabs.f32";
#if PROTEUS_ENABLE_CUDA
  if (R.Fn.getTargetModel() == TargetModelType::CUDA)
    IntrinsicName = "__nv_fabsf";
#endif

  return emitIntrinsic<float>(IntrinsicName, ResultType, RFloat);
}

template <typename T>
std::enable_if_t<is_arithmetic_unref_v<T>, Var<remove_cvref_t<T>>>
min(const Var<T> &L, const Var<T> &R) {
  FuncBase &Fn = L.Fn;
  if (&Fn != &R.Fn)
    reportFatalError("Variables should belong to the same function");

  auto ResultVar = Fn.declVar<remove_cvref_t<T>>("min_res");
  ResultVar = R;
  Fn.beginIf(L < R);
  { ResultVar = L; }
  Fn.endIf();
  return ResultVar;
}

template <typename T>
std::enable_if_t<is_arithmetic_unref_v<T>, Var<remove_cvref_t<T>>>
max(const Var<T> &L, const Var<T> &R) {

  FuncBase &Fn = L.Fn;
  if (&Fn != &R.Fn)
    reportFatalError("Variables should belong to the same function");

  auto ResultVar = Fn.declVar<remove_cvref_t<T>>("max_res");
  ResultVar = R;
  Fn.beginIf(L > R);
  { ResultVar = L; }
  Fn.endIf();
  return ResultVar;
}

} // namespace proteus

#endif // PROTEUS_FRONTEND_FUNC_H
