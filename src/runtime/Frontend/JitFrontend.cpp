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
    : TargetModel{parseTargetModel(Target)},
      TargetTriple(::proteus::getTargetTriple(TargetModel)),
      Dispatch(Dispatcher::getDispatcher(TargetModel)) {
  auto Ctx = std::make_unique<LLVMContext>();
  auto Mod = std::make_unique<Module>("JitModule", *Ctx);
  Mod->setTargetTriple(TargetTriple);
  CB = std::make_unique<LLVMCodeBuilder>(std::move(Ctx), std::move(Mod));
}

void JitModule::compile(bool Verify) {
  if (IsCompiled)
    return;

  auto &Mod = CB->getModule();

  if (Verify)
    if (verifyModule(Mod, &errs())) {
      reportFatalError("Broken module found, JIT compilation aborted!");
    }

  SmallVector<char, 0> Buffer;
  raw_svector_ostream OS(Buffer);
  WriteBitcodeToFile(Mod, OS);

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
      Dispatch.compile(CB->takeLLVMContext(), CB->takeModule(), *ModuleHash));
  IsCompiled = true;
}

void JitModule::print() { CB->getModule().print(outs(), nullptr); }

JitModule::~JitModule() = default;

const HashT &JitModule::getModuleHash() const { return *ModuleHash; }

} // namespace proteus
