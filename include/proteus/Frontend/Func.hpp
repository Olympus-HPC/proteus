#ifndef PROTEUS_FRONTEND_FUNC_HPP
#define PROTEUS_FRONTEND_FUNC_HPP

#include <deque>

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

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
  std::deque<Var> Arguments;
  std::deque<Var> Variables;
  std::deque<Var> RuntimeConstants;
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

  AllocaInst *emitAlloca(Type *Ty, StringRef Name);

  IRBuilderBase &getIRBuilder();

  Var &declVarInternal(StringRef Name, Type *Ty,
                       Type *PointerElemType = nullptr);

  template <typename T> Var &declVar(StringRef Name = "var") {
    Function *F = getFunction();
    auto *Alloca = emitAlloca(TypeMap<T>::get(F->getContext()), Name);

    return Variables.emplace_back(
        Alloca, *this, TypeMap<T>::getPointerElemType(F->getContext()));
  }

  template <typename T> Var &defVar(T Val, StringRef Name = "var") {
    Function *F = getFunction();
    auto *Alloca = emitAlloca(TypeMap<T>::get(F->getContext()), Name);

    Var &VarRef = Variables.emplace_back(
        Alloca, *this, TypeMap<T>::getPointerElemType(F->getContext()));

    VarRef = Val;

    return VarRef;
  }

  template <typename T>
  Var &defRuntimeConst(T Val, StringRef Name = "run.const.var") {
    Function *F = getFunction();
    auto *Alloca = emitAlloca(TypeMap<T>::get(F->getContext()), Name);

    Var &VarRef = RuntimeConstants.emplace_back(
        Alloca, *this, TypeMap<T>::getPointerElemType(F->getContext()));

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
          auto *Alloca = emitAlloca(TypeMap<Ts>::get(F->getContext()),
                                    "arg." + std::to_string(Arguments.size()));

          auto *Arg = F->getArg(Arguments.size());
          IRB.CreateStore(Arg, Alloca);

          Arguments.emplace_back(
              Alloca, *this, TypeMap<Ts>::getPointerElemType(F->getContext()));
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

  Var &callBuiltin(function_ref<Var &(FuncBase &)> Lower) {
    return Lower(*this);
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
