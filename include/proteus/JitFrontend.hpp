#ifndef PROTEUS_JIT_DEV_HPP
#define PROTEUS_JIT_DEV_HPP

#include <deque>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/Debug.h>
#include <llvm/TargetParser/Host.h>

#include "proteus/Error.h"
#include "proteus/JitEngineHost.hpp"

namespace proteus {
using namespace llvm;

template <typename T> struct TypeMap;

template <> struct TypeMap<double> {
  static Type *get(llvm::LLVMContext &Ctx) { return Type::getDoubleTy(Ctx); }
};

struct Func;

struct Var {
  AllocaInst *Alloca;
  Func &Fn;

  Var(AllocaInst *Alloca, Func &Fn);

  Var &operator+(Var &Other);
  Var &operator+(const double &ConstValue);

  Var &operator>(const double &ConstValue);

  Var &operator=(const Var &Other);

  Var &operator=(const double &ConstValue);
};
struct Func {
  FunctionCallee FC;
  IRBuilder<> IRB;
  IRBuilderBase::InsertPoint IP;
  std::deque<Var> Variables;
  std::vector<IRBuilderBase::InsertPoint> BlockIPs;

  Func(FunctionCallee FC);

  Function *getFunction();

  template <typename T> Var &declVar(StringRef Name) {
    Function *F = getFunction();
    auto AllocaIP = IRBuilderBase::InsertPoint(&F->getEntryBlock(),
                                               F->getEntryBlock().begin());
    IRB.restoreIP(AllocaIP);
    auto *Alloca =
        IRB.CreateAlloca(TypeMap<T>::get(F->getContext()), nullptr, Name);
    Variables.emplace_back(Alloca, *this);
    return Variables.back();
  }

  Var &arg(unsigned int ArgNo);

  void beginIf(Var &CondVar);
  void endIf();

  void ret(std::optional<std::reference_wrapper<Var>> OptRet = std::nullopt);
};

class JitModule {
private:
  std::unique_ptr<LLVMContext> Ctx;
  std::unique_ptr<Module> Mod;

  std::deque<Func> Functions;

public:
  JitModule()
      : Ctx{std::make_unique<LLVMContext>()}, Mod{std::make_unique<Module>(
                                                  "JitModule", *Ctx)} {}

  template <typename RetT, typename... ArgT> Func &addFunction(StringRef Name) {
    Mod->setTargetTriple(sys::getProcessTriple());
    FunctionCallee FC;
    if constexpr (sizeof...(ArgT) == 0)
      FC = Mod->getOrInsertFunction(Name, TypeMap<RetT>::get(*Ctx));
    else
      FC = Mod->getOrInsertFunction(Name, TypeMap<RetT>::get(*Ctx),
                                    (TypeMap<ArgT>::get(*Ctx), ...));
    Function *F = dyn_cast<Function>(FC.getCallee());
    if (!F)
      PROTEUS_FATAL_ERROR("Unexpected");
    BasicBlock *EntryBB = BasicBlock::Create(*Ctx, "entry", F);
    Functions.emplace_back(FC);
    return Functions.back();
  }

  void *compile() {
    dbgs() << "To Verify " << *Mod << "\n";
    if (verifyModule(*Mod, &errs()))
      PROTEUS_FATAL_ERROR(
          "Broken module found after optimization, JIT compilation aborted!");
    auto &Jit = JitEngineHost::instance();
    return Jit.compileJitModule(Functions.back().getFunction()->getName(),
                                std::move(Mod), std::move(Ctx));
  }

  void print() { dbgs() << "JitModule\n" << *Mod << "\n"; }
};

} // namespace proteus

#endif