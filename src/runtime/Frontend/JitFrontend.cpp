#include "proteus/JitFrontend.h"

#include "proteus/impl/CompiledLibrary.h"
#include "proteus/impl/Hashing.h"
#include "proteus/impl/JitEngineHost.h"

#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/MemoryBuffer.h>

namespace proteus {

JitModule::JitModule(const std::string &Target)
    : Ctx{std::make_unique<LLVMContext>()},
      Mod{std::make_unique<Module>("JitModule", *Ctx)},
      TargetModel{parseTargetModel(Target)},
      TargetTriple(::proteus::getTargetTriple(TargetModel)),
      Dispatch(Dispatcher::getDispatcher(TargetModel)) {
  Mod->setTargetTriple(TargetTriple);
}

void JitModule::compile(bool Verify) {
  if (IsCompiled)
    return;

  if (Verify)
    if (verifyModule(*Mod, &errs())) {
      reportFatalError("Broken module found, JIT compilation aborted!");
    }

  SmallVector<char, 0> Buffer;
  raw_svector_ostream OS(Buffer);
  WriteBitcodeToFile(*Mod, OS);

  // Create a unique module hash based on the bitcode and append to all
  // function names to make them unique.
  // TODO: Is this necessary?
  ModuleHash =
      std::make_unique<HashT>(hash(StringRef{Buffer.data(), Buffer.size()}));
  for (auto &JitF : Functions) {
    JitF->setName(JitF->getName() + ModuleHash->toMangledSuffix());
  }

  if ((Library = Dispatch.lookupCompiledLibrary(*ModuleHash))) {
    IsCompiled = true;
    return;
  }

  Library = std::make_unique<CompiledLibrary>(
      Dispatch.compile(std::move(Ctx), std::move(Mod), *ModuleHash));
  IsCompiled = true;
}

void JitModule::print() { Mod->print(outs(), nullptr); }

JitModule::~JitModule() = default;

const HashT &JitModule::getModuleHash() const { return *ModuleHash; }

} // namespace proteus
