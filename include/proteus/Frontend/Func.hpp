#ifndef PROTEUS_FRONTEND_FUNC_HPP
#define PROTEUS_FRONTEND_FUNC_HPP

#include <deque>
#include <functional>
#include <vector>

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

#include "proteus/Error.h"
#include "proteus/Frontend/Dispatcher.hpp"
#include "proteus/Frontend/TypeMap.hpp"
#include "proteus/Frontend/Var.hpp"
#include "proteus/Hashing.hpp"

namespace proteus {

struct Var;
class JitModule;
class LoopBoundsDescription;
class LoopNestBuilder;
class ForLoopBuilder;

using namespace llvm;

class FuncBase {
protected:
  JitModule &J;
  FunctionCallee FC;
  IRBuilder<> IRB;
  IRBuilderBase::InsertPoint IP;
  std::deque<Var> Arguments;
  std::deque<Var> Variables;
  std::deque<Var> RuntimeConstants;
  HashT HashValue;
  std::string Name;

  enum class ScopeKind { FUNCTION, IF, FOR, LOOP_NEST };
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
    case ScopeKind::LOOP_NEST:
      return "LOOP_NEST";
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

    HashValue = hash(HashValue, Val);

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

  template <typename RetT, typename... ArgT> void call(StringRef Name);

  Var &callBuiltin(function_ref<Var &(FuncBase &)> Lower) {
    return Lower(*this);
  }

  ForLoopBuilder ForLoop(LoopBoundsDescription Bounds);
  ForLoopBuilder ForLoop(LoopBoundsDescription Bounds,
                         std::function<void()> Body);

  LoopNestBuilder LoopNest(std::vector<ForLoopBuilder> Loops);
  LoopNestBuilder LoopNest(std::initializer_list<ForLoopBuilder> Loops);

  void ret(std::optional<std::reference_wrapper<Var>> OptRet = std::nullopt);

  StringRef getName() { return Name; }
};

template <typename RetT, typename... ArgT> class Func final : public FuncBase {
private:
  Dispatcher &Dispatch;

private:
  template <std::size_t... Is> auto getArgsImpl(std::index_sequence<Is...>) {
    return std::tie(getArg(Is)...);
  }

public:
  Func(JitModule &J, FunctionCallee FC, Dispatcher &Dispatch)
      : FuncBase(J, FC), Dispatch(Dispatch) {}

  RetT operator()(ArgT... Args);

  auto getArgs() { return getArgsImpl(std::index_sequence_for<ArgT...>{}); }
};

} // namespace proteus

#endif // PROTEUS_FRONTEND_FUNC_HPP
