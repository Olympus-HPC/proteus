#include "proteus/CppJitModule.h"
#include "proteus/TimeTracing.h"
#include "proteus/impl/CompiledLibrary.h"
#include "proteus/impl/Frontend/CppJitCompiler.h"
#include "proteus/impl/Frontend/CppJitFuncAttribute.h"

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/MemoryBuffer.h>

namespace proteus {

CppJitModule::CppJitModule(TargetModelType TargetModel, const std::string &Code,
                           const std::vector<std::string> &ExtraArgs,
                           CppJitCompilerBackend CompilerBackend)
    : TargetModel(TargetModel), Code(Code), ExtraArgs(ExtraArgs),
      CompilerBackend(CompilerBackend),
      Dispatch(Dispatcher::getDispatcher(TargetModel)) {
  if (!CppJitCompiler::isBackendSupported(this->TargetModel,
                                          this->CompilerBackend))
    reportFatalError(
        "NVCC backend is supported only for CUDA and HOST_CUDA targets");

  ModuleHash = std::make_unique<HashT>(computeCppJitModuleHash(
      this->TargetModel, this->CompilerBackend, this->Code, this->ExtraArgs));
}

CppJitModule::CppJitModule(const std::string &Target, const std::string &Code,
                           const std::vector<std::string> &ExtraArgs,
                           CppJitCompilerBackend CompilerBackend)
    : CppJitModule(parseTargetModel(Target), Code, ExtraArgs, CompilerBackend) {
}

CppJitModule::~CppJitModule() = default;

void CppJitModule::compile() {
  TIMESCOPE(CppJitModule, compile);

  if ((Library = Dispatch.lookupCompiledLibrary(*ModuleHash))) {
    IsCompiled = true;
    return;
  }

  CppJitCompileRequest Request{TargetModel,
                               CompilerBackend,
                               Code,
                               ExtraArgs,
                               *ModuleHash,
                               TargetModel != TargetModelType::HOST
                                   ? Dispatch.getDeviceArch().str()
                                   : std::string{}};
  auto Compiler = createCppJitCompiler(Request);
  auto Artifact = Compiler->compile(Request);

  switch (Artifact.ArtifactKind) {
  case CppJitArtifact::Kind::SharedLibrary:
    Dispatch.registerDynamicLibrary(*ModuleHash, Artifact.Path.c_str());
    Library = Dispatch.lookupCompiledLibrary(*ModuleHash);
    if (!Library)
      reportFatalError("Expected non-null library after compilation");
    break;
  case CppJitArtifact::Kind::DeviceBinary:
    Dispatch.registerObject(*ModuleHash,
                            Artifact.ObjectBuffer->getMemBufferRef());
    Library = Dispatch.lookupCompiledLibrary(*ModuleHash);
    if (!Library)
      reportFatalError("Expected non-null library after NVCC compilation");
    break;
  case CppJitArtifact::Kind::LLVMIR: {
    auto ObjectModule = Dispatch.compile(std::move(Artifact.Ctx),
                                         std::move(Artifact.Mod), *ModuleHash,
                                         /*DisableIROpt=*/true);
    Library = std::make_unique<CompiledLibrary>(std::move(ObjectModule));
    break;
  }
  }

  IsCompiled = true;
}

void CppJitModule::setFuncAttribute(void *KernelFunc, CppJitFuncAttribute Attr,
                                    int Value) {
  proteus::setFuncAttribute(TargetModel, KernelFunc, Attr, Value);
}

void *CppJitModule::getFunctionAddress(const std::string &Name) {
  TIMESCOPE(CppJitModule, getFunctionAddress);
  return Dispatch.getFunctionAddress(Name, *ModuleHash, getLibrary());
}

DispatchResult CppJitModule::launch(void *KernelFunc, LaunchDims GridDim,
                                    LaunchDims BlockDim, void *KernelArgs[],
                                    uint64_t ShmemSize, void *Stream) {
  TIMESCOPE(CppJitModule, launch);
  return Dispatch.launch(KernelFunc, GridDim, BlockDim, KernelArgs, ShmemSize,
                         Stream);
}

} // namespace proteus
