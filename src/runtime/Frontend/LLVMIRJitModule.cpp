#include "proteus/LLVMIRJitModule.h"

#include "proteus/TimeTracing.h"
#include "proteus/impl/CompiledLibrary.h"
#include "proteus/impl/CoreLLVM.h"
#include "proteus/impl/Frontend/JitFuncAttribute.h"
#include "proteus/impl/Hashing.h"

#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/TargetParser/Host.h>

namespace proteus {
namespace {

std::unique_ptr<llvm::Module> parseLLVMIRModule(const std::string &Code,
                                                LLVMIRInputKind InputKind,
                                                llvm::LLVMContext &Context) {
  llvm::MemoryBufferRef Buffer(Code, "LLVMIRJitModule");

  auto ParseBitcode = [&]() -> std::unique_ptr<llvm::Module> {
    auto Parsed = llvm::parseBitcodeFile(Buffer, Context);
    if (!Parsed) {
      llvm::consumeError(Parsed.takeError());
      return nullptr;
    }
    return std::move(*Parsed);
  };

  if (InputKind == LLVMIRInputKind::Bitcode) {
    auto Parsed = llvm::parseBitcodeFile(Buffer, Context);
    if (!Parsed)
      reportFatalError("LLVMIRJitModule: failed to parse bitcode: " +
                       llvm::toString(Parsed.takeError()));
    return std::move(*Parsed);
  }

  if (InputKind == LLVMIRInputKind::Auto) {
    if (auto ParsedBitcode = ParseBitcode()) {
      return ParsedBitcode;
    }
  }

  llvm::SMDiagnostic Diag;
  auto ParsedIR = llvm::parseIR(Buffer, Diag, Context);
  if (!ParsedIR) {
    reportFatalError("LLVMIRJitModule: failed to parse LLVM IR: " +
                     Diag.getMessage().str());
  }

  return ParsedIR;
}

} // namespace

LLVMIRJitModule::LLVMIRJitModule(TargetModelType TargetModel,
                                 const std::string &Code,
                                 LLVMIRInputKind InputKind)
    : TargetModel(TargetModel), Code(Code), InputKind(InputKind),
      Dispatch(Dispatcher::getDispatcher(TargetModel)) {
  // Hash the raw input, declared input kind, target model, and codegen config
  // because each can change the compiled artifact.
  ModuleHash = std::make_unique<HashT>(hash(static_cast<int>(TargetModel),
                                            static_cast<int>(InputKind), Code,
                                            Config::get().getCGConfig()));
}

LLVMIRJitModule::LLVMIRJitModule(const std::string &Target,
                                 const std::string &Code,
                                 LLVMIRInputKind InputKind)
    : LLVMIRJitModule(parseTargetModel(Target), Code, InputKind) {}

LLVMIRJitModule::~LLVMIRJitModule() = default;

void LLVMIRJitModule::compile(bool Verify) {
  TIMESCOPE(LLVMIRJitModule, compile);

  if (IsCompiled)
    return;

  if ((Library = Dispatch.lookupCompiledLibrary(*ModuleHash))) {
    IsCompiled = true;
    return;
  }

  auto Context = std::make_unique<llvm::LLVMContext>();
  auto Module = parseLLVMIRModule(Code, InputKind, *Context);
  if (!Module)
    reportFatalError("LLVMIRJitModule: expected non-null parsed LLVM module");

  if (Module->getTargetTriple().empty())
    Module->setTargetTriple(getTargetTriple(TargetModel));

  if (Module->getDataLayout().isDefault()) {
    const llvm::StringRef Arch = isHostTargetModel(TargetModel)
                                     ? llvm::sys::getHostCPUName()
                                     : Dispatch.getDeviceArch();
    auto TMExpected = detail::createTargetMachine(*Module, Arch);
    if (!TMExpected)
      reportFatalError("LLVMIRJitModule: failed to create target machine");
    Module->setDataLayout((*TMExpected)->createDataLayout());
  }

  if (Verify && verifyModule(*Module, &llvm::errs()))
    reportFatalError("Broken module found, JIT compilation aborted!");

  Library = std::make_unique<CompiledLibrary>(
      Dispatch.compile(std::move(Context), std::move(Module), *ModuleHash));
  IsCompiled = true;
}

void LLVMIRJitModule::setFuncAttribute(void *KernelFunc, JitFuncAttribute Attr,
                                       int Value) {
  proteus::setFuncAttribute(TargetModel, KernelFunc, Attr, Value);
}

void *LLVMIRJitModule::getFunctionAddress(const std::string &Name) {
  TIMESCOPE(LLVMIRJitModule, getFunctionAddress);
  if (!ModuleHash)
    reportFatalError("LLVMIRJitModule: expected module hash after compilation");
  return Dispatch.getFunctionAddress(Name, *ModuleHash, getLibrary());
}

DispatchResult LLVMIRJitModule::launch(void *KernelFunc, LaunchDims GridDim,
                                       LaunchDims BlockDim, void *KernelArgs[],
                                       uint64_t ShmemSize, void *Stream) {
  TIMESCOPE(LLVMIRJitModule, launch);
  return Dispatch.launch(KernelFunc, GridDim, BlockDim, KernelArgs, ShmemSize,
                         Stream);
}

} // namespace proteus
