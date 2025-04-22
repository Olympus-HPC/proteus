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
#include "proteus/frontend/Func.hpp"
#include "proteus/frontend/TypeMap.hpp"

namespace proteus {
using namespace llvm;

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
    FC = Mod->getOrInsertFunction(Name, TypeMap<RetT>::get(*Ctx),
                                  TypeMap<ArgT>::get(*Ctx)...);
    Function *F = dyn_cast<Function>(FC.getCallee());
    if (!F)
      PROTEUS_FATAL_ERROR("Unexpected");
    auto &Fn = Functions.emplace_back(FC);

    Fn.declArgs<ArgT...>();
    return Fn;
  }

  void *compile() {
    if (verifyModule(*Mod, &errs()))
      PROTEUS_FATAL_ERROR(
          "Broken module found after optimization, JIT compilation aborted!");
    auto &Jit = JitEngineHost::instance();
    return Jit.compileJitModule(Functions.back().getFunction()->getName(),
                                std::move(Mod), std::move(Ctx));
  }

  void print() { Mod->print(outs(), nullptr); }
};

} // namespace proteus

#endif
