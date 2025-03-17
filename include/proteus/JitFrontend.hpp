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

template <typename T> struct LLVMTypeMap;

template <> struct LLVMTypeMap<double> {
  static Type *get(llvm::LLVMContext &Ctx) { return Type::getDoubleTy(Ctx); }
};

class JitModule {

private:
  struct Func;

  struct Var {
    AllocaInst *Alloca;
    Func &Fn;

    Var(AllocaInst *Alloca, Func &Fn) : Alloca(Alloca), Fn(Fn) {}

    Var &operator+(Var &Other) {
      Function *F = Fn.getFunction();
      IRBuilder IRB{&F->getEntryBlock()};
      auto *ResultAlloca =
          IRB.CreateAlloca(Alloca->getAllocatedType(), nullptr, "res");
      IRB.SetInsertPoint(&F->back());
      auto *LHS = IRB.CreateLoad(Alloca->getAllocatedType(), Alloca);
      auto *RHS =
          IRB.CreateLoad(Other.Alloca->getAllocatedType(), Other.Alloca);
      auto *Result = IRB.CreateFAdd(LHS, RHS);
      IRB.CreateStore(Result, ResultAlloca);

      Fn.Variables.emplace_back(ResultAlloca, Fn);
      return Fn.Variables.back();
    }

    Var &operator=(const Var &Other) {
      Function *F = Fn.getFunction();
      IRBuilder IRB{&F->back()};
      auto *RHS =
          IRB.CreateLoad(Other.Alloca->getAllocatedType(), Other.Alloca);
      IRB.CreateStore(RHS, Alloca);
      return *this;
    }

    Var &operator=(const double &ConstValue) {
      Function *F = Fn.getFunction();
      IRBuilder IRB{&F->back()};
      IRB.CreateStore(ConstantFP::get(Alloca->getAllocatedType(), ConstValue),
                      Alloca);
      return *this;
    }
  };

  struct Func {
    FunctionCallee FC;
    std::deque<Var> Variables;

    Func(FunctionCallee FC) : FC(FC) {}

    Function *getFunction() {
      Function *F = dyn_cast<Function>(FC.getCallee());
      if (!F)
        PROTEUS_FATAL_ERROR("Expected LLVM Function");
      return F;
    }

    void
    addRet(std::optional<std::reference_wrapper<Var>> OptRet = std::nullopt) {
      IRBuilder IRB{&getFunction()->back()};
      if (OptRet == std::nullopt) {
        IRB.CreateRetVoid();
        return;
      }

      auto *RetAlloca = OptRet->get().Alloca;
      auto *Ret = IRB.CreateLoad(RetAlloca->getAllocatedType(), RetAlloca);
      IRB.CreateRet(Ret);
    }
  };

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
      FC = Mod->getOrInsertFunction(Name, RetT::get(*Ctx));
    else
      FC = Mod->getOrInsertFunction(Name, RetT::get(*Ctx),
                                    (ArgT::get(*Ctx), ...));
    Function *F = dyn_cast<Function>(FC.getCallee());
    if (!F)
      PROTEUS_FATAL_ERROR("Unexpected");
    BasicBlock *EntryBB = BasicBlock::Create(*Ctx, "entry", F);
    Functions.emplace_back(FC);
    return Functions.back();
  }

  template <typename VarT> Var &addVariable(StringRef Name, Func &Fn) {
    Function *F = Fn.getFunction();
    IRBuilder IRB{&F->getEntryBlock()};
    auto *Alloca = IRB.CreateAlloca(VarT::get(*Ctx), nullptr, Name);
    Fn.Variables.emplace_back(Alloca, Fn);
    return Fn.Variables.back();
  }

  void *compile() {
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