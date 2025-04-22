#ifndef PROTEUS_FRONTEND_FUNC_HPP
#define PROTEUS_FRONTEND_FUNC_HPP

#include <deque>

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

#include "proteus/frontend/TypeMap.hpp"
#include "proteus/frontend/Var.hpp"

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
  std::vector<IRBuilderBase::InsertPoint> BlockIPs;
  std::string Name;

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

  void beginFunction();
  void endFunction();

  void beginIf(Var &CondVar);
  void endIf();

  void beginLoop(Var &IterVar, Var &InitVar, Var &UpperBound, Var &IncVar);
  void endLoop();

  template <typename RetT, typename... ArgT> void call(StringRef Name);

  void ret(std::optional<std::reference_wrapper<Var>> OptRet = std::nullopt);
};

} // namespace proteus

#endif // PROTEUS_FRONTEND_FUNC_HPP
