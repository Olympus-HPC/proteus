#ifndef PROTEUS_FRONTEND_FUNC_HPP
#define PROTEUS_FRONTEND_FUNC_HPP

#include <deque>

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

#include "proteus/Error.h"
#include "proteus/Frontend/TypeMap.hpp"
#include "proteus/Frontend/Var.hpp"

namespace proteus {

struct Var;

using namespace llvm;

class Func {
private:
  FunctionCallee FC;
  IRBuilder<> IRB;
  IRBuilderBase::InsertPoint IP;
  std::deque<Var> Arguments;
  std::deque<Var> Variables;
  std::vector<IRBuilderBase::InsertPoint> ContIPs;
  std::string Name;

  enum class ScopeKind { FUNCTION, IF, LOOP };
  struct Scope {
    std::string File;
    int Line;
    ScopeKind Kind;

    explicit Scope(const char *File, int Line, ScopeKind Kind)
        : File(File), Line(Line), Kind(Kind) {}
  };
  std::vector<Scope> Scopes;

  std::string toString(ScopeKind Kind) {
    switch (Kind) {
    case ScopeKind::FUNCTION:
      return "FUNCTION";
    case ScopeKind::IF:
      return "IF";
    case ScopeKind::LOOP:
      return "LOOP";
    defaut:
      PROTEUS_FATAL_ERROR("Unsupported Kind " +
                          std::to_string(static_cast<int>(Kind)));
    }
  }

public:
  Func(FunctionCallee FC);

  Function *getFunction();

  AllocaInst *emitAlloca(Type *Ty, StringRef Name);

  IRBuilderBase &getIRB();

  Var &declVarInternal(StringRef Name, Type *Ty,
                       Type *PointerElemType = nullptr);

  template <typename T> Var &declVar(StringRef Name) {
    Function *F = getFunction();
    auto *Alloca = emitAlloca(TypeMap<T>::get(F->getContext()), Name);

    return Variables.emplace_back(
        Alloca, *this, TypeMap<T>::getPointerElemType(F->getContext()));
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

  void beginLoop(Var &IterVar, Var &InitVar, Var &UpperBound, Var &IncVar,
                 const char *File = __builtin_FILE(),
                 int Line = __builtin_LINE());
  void endLoop();

  template <typename RetT, typename... ArgT> void call(StringRef Name);

  Var &callBuiltin(function_ref<Var &(Func &)> Lower) { return Lower(*this); }

  void ret(std::optional<std::reference_wrapper<Var>> OptRet = std::nullopt);

  StringRef getName() { return Name; }
};

} // namespace proteus

#endif // PROTEUS_FRONTEND_FUNC_HPP
