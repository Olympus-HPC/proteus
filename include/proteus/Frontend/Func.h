#ifndef PROTEUS_FRONTEND_FUNC_H
#define PROTEUS_FRONTEND_FUNC_H

#include "proteus/AddressSpace.h"
#include "proteus/Error.h"
#include "proteus/Frontend/Dispatcher.h"
#include "proteus/Frontend/LLVMCodeBuilder.h"
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
class Function;
} // namespace llvm

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

  /// Get the underlying LLVMCodeBuilder for direct IR generation.
  LLVMCodeBuilder &getCodeBuilder();

protected:
  JitModule &J;

  std::string Name;
  LLVMCodeBuilder *CB;
  llvm::Function *LLVMFunc;

public:
  FuncBase(JitModule &J, LLVMCodeBuilder &CB, const std::string &Name,
           IRType RetTy, const std::vector<IRType> &ArgTys);
  ~FuncBase();

  llvm::Function *getFunction();
  IRValue getArg(size_t Idx);

#if PROTEUS_ENABLE_CUDA || PROTEUS_ENABLE_HIP
  // Kernel management.
  void setKernel();
  void setLaunchBoundsForKernel(int MaxThreadsPerBlock, int MinBlocksPerSM);
#endif

  template <typename T> Var<T> declVar(const std::string &Name = "var") {
    static_assert(!std::is_array_v<T>, "Expected non-array type");
    static_assert(!std::is_reference_v<T>,
                  "declVar does not support reference types");

    if constexpr (std::is_pointer_v<T>) {
      IRType ElemIRTy = *TypeMap<T>::getPointerElemType();
      return Var<T>{CB->createPointerStorage(Name, ElemIRTy), *CB};
    } else {
      IRType AllocaIRTy = TypeMap<T>::get();
      return Var<T>{CB->createScalarStorage(Name, AllocaIRTy), *CB};
    }
  }

  template <typename T>
  Var<T> declVar(size_t NElem, AddressSpace AS = AddressSpace::DEFAULT,
                 const std::string &Name = "array_var") {
    static_assert(std::is_array_v<T>, "Expected array type");

    IRType ArrIRTy = TypeMap<T>::get(NElem);
    return Var<T>{CB->createArrayStorage(Name, AS, ArrIRTy), *CB};
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
  // Note: prefer calling var.convert<U>() directly on the Var.
  template <typename U, typename T> auto convert(const Var<T> &V) {
    return V.template convert<U>();
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
      : FuncBase(J, CB, Name, TypeMap<RetT>::get(), {TypeMap<ArgT>::get()...}),
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

  CB->beginFor(IterVar.getSlot(), IterVar.getValueType(), Init.loadValue(),
               UpperBound.loadValue(), Inc.loadValue(),
               std::is_signed_v<std::remove_const_t<IterT>>, File, Line);
}

template <typename CondLambda>
void FuncBase::beginWhile(CondLambda &&Cond, const char *File, int Line) {
  CB->beginWhile([Cond = std::forward<CondLambda>(
                      Cond)]() -> IRValue { return Cond().loadValue(); },
                 File, Line);
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

  std::vector<IRType> ArgTys = unpackArgTypes(ArgT{});
  getCodeBuilder().createCall(Name, TypeMap<RetT>::get(), ArgTys,
                              {GetArgVal(ArgsVars)...});
}

template <typename Sig>
std::enable_if_t<!std::is_void_v<typename FnSig<Sig>::RetT>,
                 Var<typename FnSig<Sig>::RetT>>
FuncBase::call(const std::string &Name) {
  using RetT = typename FnSig<Sig>::RetT;

  auto Call = getCodeBuilder().createCall(Name, TypeMap<RetT>::get());
  Var<RetT> Ret = declVar<RetT>("ret");
  Ret.storeValue(Call);
  return Ret;
}

template <typename Sig>
std::enable_if_t<std::is_void_v<typename FnSig<Sig>::RetT>, void>
FuncBase::call(const std::string &Name) {
  using RetT = typename FnSig<Sig>::RetT;
  getCodeBuilder().createCall(Name, TypeMap<RetT>::get());
}

template <typename... Ts>
std::vector<IRType> unpackArgTypes(ArgTypeList<Ts...>) {
  return {TypeMap<Ts>::get()...};
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

  std::vector<IRType> ArgTys = unpackArgTypes(ArgT{});
  std::vector<IRValue> ArgVals = {GetArgVal(ArgsVars)...};

  auto Call =
      getCodeBuilder().createCall(Name, TypeMap<RetT>::get(), ArgTys, ArgVals);

  Var<RetT> Ret = declVar<RetT>("ret");
  Ret.storeValue(Call);
  return Ret;
}

template <typename T>
std::enable_if_t<is_arithmetic_unref_v<T>, Var<T>>
FuncBase::atomicAdd(const Var<T *> &Addr, const Var<T> &Val) {
  static_assert(std::is_arithmetic_v<T>, "atomicAdd requires arithmetic type");

  IRValue Result = CB->createAtomicAdd(Addr.loadPointer(), Val.loadValue());
  auto Ret = declVar<T>("atomic.add.res.");
  Ret.storeValue(Result);
  return Ret;
}

template <typename T>
std::enable_if_t<is_arithmetic_unref_v<T>, Var<T>>
FuncBase::atomicSub(const Var<T *> &Addr, const Var<T> &Val) {
  static_assert(std::is_arithmetic_v<T>, "atomicSub requires arithmetic type");

  IRValue Result = CB->createAtomicSub(Addr.loadPointer(), Val.loadValue());
  auto Ret = declVar<T>("atomic.sub.res.");
  Ret.storeValue(Result);
  return Ret;
}

template <typename T>
std::enable_if_t<is_arithmetic_unref_v<T>, Var<T>>
FuncBase::atomicMax(const Var<T *> &Addr, const Var<T> &Val) {
  static_assert(std::is_arithmetic_v<T>, "atomicMax requires arithmetic type");

  IRValue Result = CB->createAtomicMax(Addr.loadPointer(), Val.loadValue());
  auto Ret = declVar<T>("atomic.max.res.");
  Ret.storeValue(Result);
  return Ret;
}

template <typename T>
std::enable_if_t<is_arithmetic_unref_v<T>, Var<T>>
FuncBase::atomicMin(const Var<T *> &Addr, const Var<T> &Val) {
  static_assert(std::is_arithmetic_v<T>, "atomicMin requires arithmetic type");

  IRValue Result = CB->createAtomicMin(Addr.loadPointer(), Val.loadValue());
  auto Ret = declVar<T>("atomic.min.res.");
  Ret.storeValue(Result);
  return Ret;
}

inline void FuncBase::ret() { CB->createRetVoid(); }

template <typename T> void FuncBase::ret(const Var<T> &RetVal) {
  IRValue RetValue = RetVal.loadValue();
  CB->createRet(RetValue);
}
} // namespace proteus

#endif // PROTEUS_FRONTEND_FUNC_H
