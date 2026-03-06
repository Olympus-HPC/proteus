#include "proteus/JitFrontend.h"

#include "proteus/impl/CompiledLibrary.h"
#include "proteus/impl/Hashing.h"

#if defined(PROTEUS_ENABLE_MLIR)
#include "proteus/Frontend/MLIRCodeBuilder.h"
#endif

#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/MemoryBuffer.h>

namespace proteus {

JitModule::JitModule(const std::string &Target, const std::string &Backend)
    : TargetModel{parseTargetModel(Target)},
      Dispatch(Dispatcher::getDispatcher(TargetModel)) {
  if (Backend == "llvm") {
    auto Ctx = std::make_unique<LLVMContext>();
    auto Mod = std::make_unique<Module>("JitModule", *Ctx);
    CB = std::make_unique<LLVMCodeBuilder>(std::move(Ctx), std::move(Mod),
                                           TargetModel);
  } else if (Backend == "mlir") {
#if defined(PROTEUS_ENABLE_MLIR)
    CB = std::make_unique<MLIRCodeBuilder>(TargetModel);
#else
    reportFatalError(
        "MLIR backend not enabled. Rebuild with -DPROTEUS_ENABLE_MLIR=ON");
#endif
  } else {
    reportFatalError("Unsupported backend: " + Backend);
  }
}

void JitModule::compile(bool Verify) {
  if (IsCompiled)
    return;

  std::unique_ptr<LLVMContext> Ctx;
  std::unique_ptr<Module> Mod;

  if (CB->getBackendKind() == CodeBuilderKind::LLVM) {
    auto *LCB = static_cast<LLVMCodeBuilder *>(CB.get());
    Ctx = LCB->takeLLVMContext();
    Mod = LCB->takeModule();
  }

#if defined(PROTEUS_ENABLE_MLIR)
  else if (CB->getBackendKind() == CodeBuilderKind::MLIR) {
    // MLIRCodeBuilder lowers to host LLVM IR for HOST, and device LLVM IR for
    // CUDA/HIP; dispatcher compilation/launch flow remains unchanged.
    auto *MLIRCB = static_cast<MLIRCodeBuilder *>(CB.get());
    if (!isHostTargetModel(TargetModel))
      MLIRCB->setDeviceArch(Dispatch.getDeviceArch().str());
    Ctx = MLIRCB->takeContext();
    Mod = MLIRCB->takeModule();
  }
#endif
  else {
    reportFatalError("compile() not supported for this backend");
  }

  if (!Ctx || !Mod)
    reportFatalError("compile() expected non-null LLVM context/module");

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
    const std::string OldName = JitF->getName();
    const std::string NewName = OldName + ModuleHash->toMangledSuffix();

    auto *Fn = Mod->getFunction(OldName);

    if (!Fn)
      reportFatalError("compile() failed to find function in LLVM module: " +
                       OldName);

    Fn->setName(NewName);
    JitF->setFrontendName(NewName);
  }

  if ((Library = Dispatch.lookupCompiledLibrary(*ModuleHash))) {
    IsCompiled = true;
    return;
  }

  Library = std::make_unique<CompiledLibrary>(
      Dispatch.compile(std::move(Ctx), std::move(Mod), *ModuleHash));
  IsCompiled = true;
}

void JitModule::print() {
  if (CB->getBackendKind() == CodeBuilderKind::LLVM) {
    static_cast<LLVMCodeBuilder *>(CB.get())->getModule().print(outs(),
                                                                nullptr);
    return;
  }

#if defined(PROTEUS_ENABLE_MLIR)
  if (CB->getBackendKind() == CodeBuilderKind::MLIR) {
    static_cast<MLIRCodeBuilder *>(CB.get())->print();
    return;
  }
#endif

  reportFatalError("print() not supported for this backend");
}

void JitModule::printLLVMIR() {
  if (CB->getBackendKind() == CodeBuilderKind::LLVM) {
    static_cast<LLVMCodeBuilder *>(CB.get())->getModule().print(outs(),
                                                                nullptr);
    return;
  }

#if defined(PROTEUS_ENABLE_MLIR)
  if (CB->getBackendKind() == CodeBuilderKind::MLIR) {
    auto *MLIRCB = static_cast<MLIRCodeBuilder *>(CB.get());
    if (!isHostTargetModel(TargetModel))
      MLIRCB->setDeviceArch(Dispatch.getDeviceArch().str());
    MLIRCB->printLLVMIR(outs());
    return;
  }
#endif

  reportFatalError("printLLVMIR() not supported for this backend");
}

JitModule::~JitModule() = default;

const HashT &JitModule::getModuleHash() const { return *ModuleHash; }

} // namespace proteus
