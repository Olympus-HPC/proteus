#include "proteus/MLIRJitModule.h"

#include "proteus/TimeTracing.h"
#include "proteus/impl/CompiledLibrary.h"
#include "proteus/impl/Frontend/CppJitFuncAttribute.h"
#include "proteus/impl/Frontend/MLIRLower.h"
#include "proteus/impl/Hashing.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

namespace proteus {

MLIRJitModule::MLIRJitModule(TargetModelType TargetModel,
                             const std::string &Code)
    : TargetModel(TargetModel), Code(Code),
      Dispatch(Dispatcher::getDispatcher(TargetModel)) {}

MLIRJitModule::MLIRJitModule(const std::string &Target, const std::string &Code)
    : MLIRJitModule(parseTargetModel(Target), Code) {}

MLIRJitModule::~MLIRJitModule() = default;

void MLIRJitModule::compile(bool Verify) {
  TIMESCOPE(MLIRJitModule, compile);

  if (IsCompiled)
    return;

  mlir::MLIRContext Context;
  loadMLIRLoweringDialects(Context);

  mlir::OwningOpRef<mlir::ModuleOp> ParsedModule =
      mlir::parseSourceString<mlir::ModuleOp>(Code, &Context);
  if (!ParsedModule)
    reportFatalError("MLIRJitModule: failed to parse MLIR source");

  MLIRLoweringOptions Options;
  Options.TargetModel = TargetModel;
  Options.DiagnosticPrefix = "MLIRJitModule";
  if (TargetModel != TargetModelType::HOST)
    Options.DeviceArch = Dispatch.getDeviceArch().str();

  MLIRLoweringResult Lowered = lowerMLIRModuleToLLVM(*ParsedModule, Options);
  if (!Lowered.Ctx || !Lowered.Mod)
    reportFatalError("MLIRJitModule: expected non-null lowered LLVM module");

  if (Verify && verifyModule(*Lowered.Mod, &llvm::errs()))
    reportFatalError("Broken module found, JIT compilation aborted!");

  llvm::SmallVector<char, 0> Buffer;
  llvm::raw_svector_ostream OS(Buffer);
  llvm::WriteBitcodeToFile(*Lowered.Mod, OS);
  ModuleHash = std::make_unique<HashT>(
      hash(llvm::StringRef{Buffer.data(), Buffer.size()}));

  if ((Library = Dispatch.lookupCompiledLibrary(*ModuleHash))) {
    IsCompiled = true;
    return;
  }

  Library = std::make_unique<CompiledLibrary>(Dispatch.compile(
      std::move(Lowered.Ctx), std::move(Lowered.Mod), *ModuleHash));
  IsCompiled = true;
}

void MLIRJitModule::setFuncAttribute(void *KernelFunc, CppJitFuncAttribute Attr,
                                     int Value) {
  proteus::setFuncAttribute(TargetModel, KernelFunc, Attr, Value);
}

void *MLIRJitModule::getFunctionAddress(const std::string &Name) {
  TIMESCOPE(MLIRJitModule, getFunctionAddress);
  if (!ModuleHash)
    reportFatalError("MLIRJitModule: expected module hash after compilation");
  return Dispatch.getFunctionAddress(Name, *ModuleHash, getLibrary());
}

DispatchResult MLIRJitModule::launch(void *KernelFunc, LaunchDims GridDim,
                                     LaunchDims BlockDim, void *KernelArgs[],
                                     uint64_t ShmemSize, void *Stream) {
  TIMESCOPE(MLIRJitModule, launch);
  return Dispatch.launch(KernelFunc, GridDim, BlockDim, KernelArgs, ShmemSize,
                         Stream);
}

} // namespace proteus
