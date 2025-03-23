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

  static Type *getPointerElemType(llvm::LLVMContext &Ctx) { return nullptr; }
};

template <> struct TypeMap<double *> {
  static Type *get(llvm::LLVMContext &Ctx) {
    return PointerType::getUnqual(Ctx);
  }

  static Type *getPointerElemType(llvm::LLVMContext &Ctx) {
    return Type::getDoubleTy(Ctx);
  }
};

struct Func;

struct Var {
  AllocaInst *Alloca;
  Type *PointerElemType;
  Func &Fn;

  Var(AllocaInst *Alloca, Func &Fn, Type *PointerElemType = nullptr);

  Var &operator+(Var &Other);
  Var &operator+(const double &ConstValue);

  Var &operator>(const double &ConstValue);

  Var &operator=(const Var &Other);

  Var &operator=(const double &ConstValue);
  Var &operator[](size_t I);
};
struct Func {
  FunctionCallee FC;
  IRBuilder<> IRB;
  IRBuilderBase::InsertPoint IP;
  std::deque<Var> Arguments;
  std::deque<Var> Variables;
  std::vector<IRBuilderBase::InsertPoint> BlockIPs;

  Func(FunctionCallee FC);

  Function *getFunction();

  AllocaInst *emitAlloca(Type *Ty, StringRef Name);

  template <typename T> Var &declVar(StringRef Name) {
    Function *F = getFunction();
    auto *Alloca = emitAlloca(TypeMap<T>::get(F->getContext()), Name);

    auto &Var = Variables.emplace_back(
        Alloca, *this, TypeMap<T>::getPointerElemType(F->getContext()));
    return Var;
  }

  template <typename T> Var &declArg(StringRef Name) {
    Function *F = getFunction();
    auto *Alloca = emitAlloca(TypeMap<T>::get(F->getContext()), Name);

    auto *Arg = F->getArg(Arguments.size());
    IRB.CreateStore(Arg, Alloca);
    auto &ArgVar = Arguments.emplace_back(
        Alloca, *this, TypeMap<T>::getPointerElemType(F->getContext()));
    return ArgVar;
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
    auto &Fn = Functions.emplace_back(FC);
    (Fn.declArg<ArgT>("arg"), ...);
    return Fn;
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