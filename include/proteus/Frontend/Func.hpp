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
#include "proteus/Frontend/Var.hpp"

namespace proteus {

struct Var;
class JitModule;
class LoopBoundInfo;
template <typename... ForLoopBuilders> class LoopNestBuilder;
template <typename BodyLambda> class ForLoopBuilder;

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

  void beginFor(Var &IterVar, Var &InitVar, Var &UpperBound, Var &IncVar,
                const char *File = __builtin_FILE(),
                int Line = __builtin_LINE());
  void endFor();

  template <typename RetT, typename... ArgT>
  std::enable_if_t<!std::is_void_v<RetT>, Var &> call(StringRef Name);
  template <typename RetT, typename... ArgT>
  std::enable_if_t<std::is_void_v<RetT>, void> call(StringRef Name);

  template <typename RetT, typename... ArgVarTs>
  std::enable_if_t<!std::is_void_v<RetT>, Var &> call(StringRef Name,
                                                      ArgVarTs &...Args);
  template <typename RetT, typename... ArgVarTs>
  std::enable_if_t<std::is_void_v<RetT>, void> call(StringRef Name,
                                                    ArgVarTs &...Args);

  template <typename BuiltinFuncT>
  decltype(auto) callBuiltin(BuiltinFuncT &&BuiltinFunc) {
    using RetT = std::invoke_result_t<BuiltinFuncT &, FuncBase &>;
    if constexpr (std::is_void_v<RetT>) {
      std::invoke(std::forward<BuiltinFuncT>(BuiltinFunc), *this);
    } else {
      return std::invoke(std::forward<BuiltinFuncT>(BuiltinFunc), *this);
    }
  }

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
};

template <typename RetT, typename... ArgT> class Func final : public FuncBase {
private:
  Dispatcher &Dispatch;
  RetT (*CompiledFunc)(ArgT...) = nullptr;

private:
  template <std::size_t... Is> auto getArgsImpl(std::index_sequence<Is...>) {
    return std::tie(getArg(Is)...);
  }

public:
  Func(JitModule &J, FunctionCallee FC, Dispatcher &Dispatch)
      : FuncBase(J, FC), Dispatch(Dispatch) {}

  RetT operator()(ArgT... Args);

  auto getArgs() { return getArgsImpl(std::index_sequence_for<ArgT...>{}); }

  auto getCompiledFunc() const { return CompiledFunc; }

  void setCompiledFunc(RetT (*CompiledFuncIn)(ArgT...)) {
    CompiledFunc = CompiledFuncIn;
  }
};

} // namespace proteus

#endif // PROTEUS_FRONTEND_FUNC_HPP
