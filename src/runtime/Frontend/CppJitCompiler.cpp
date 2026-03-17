#include "proteus/impl/Frontend/CppJitCompiler.h"

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>

namespace proteus {

bool CppJitCompiler::isBackendSupported(TargetModelType TM,
                                        CppJitCompilerBackend Backend) {
  if (Backend == CppJitCompilerBackend::Clang)
    return true;

  return TM == TargetModelType::CUDA || TM == TargetModelType::HOST_CUDA;
}

CppJitArtifact::CppJitArtifact() : ArtifactKind(Kind::LLVMIR) {}

CppJitArtifact::CppJitArtifact(CppJitArtifact &&) = default;

CppJitArtifact &CppJitArtifact::operator=(CppJitArtifact &&) = default;

CppJitArtifact::~CppJitArtifact() {
  if (ArtifactKind == Kind::SharedLibrary && !Path.empty())
    llvm::sys::fs::remove(Path);
}

CppJitArtifact CppJitArtifact::sharedLibrary(std::string Path) {
  CppJitArtifact Artifact;
  Artifact.ArtifactKind = Kind::SharedLibrary;
  Artifact.Path = std::move(Path);
  return Artifact;
}

CppJitArtifact
CppJitArtifact::deviceBinary(std::unique_ptr<llvm::MemoryBuffer> Obj) {
  CppJitArtifact Artifact;
  Artifact.ArtifactKind = Kind::DeviceBinary;
  Artifact.ObjectBuffer = std::move(Obj);
  return Artifact;
}

CppJitArtifact CppJitArtifact::llvmIR(std::unique_ptr<llvm::LLVMContext> Ctx,
                                      std::unique_ptr<llvm::Module> Mod) {
  CppJitArtifact Artifact;
  Artifact.ArtifactKind = Kind::LLVMIR;
  Artifact.Ctx = std::move(Ctx);
  Artifact.Mod = std::move(Mod);
  return Artifact;
}

HashT computeCppJitModuleHash(TargetModelType TM, CppJitCompilerBackend Backend,
                              const std::string &Code,
                              const std::vector<std::string> &ExtraArgs) {
  HashT H = hash(static_cast<int>(TM), static_cast<int>(Backend), Code);
  for (const auto &Arg : ExtraArgs)
    H = hashCombine(H, hash(Arg));
  return H;
}

std::unique_ptr<CppJitCompiler>
createCppJitCompiler(const CppJitCompileRequest &Request) {
  switch (Request.Backend) {
  case CppJitCompilerBackend::Clang:
    return createCppJitCompilerClang();
  case CppJitCompilerBackend::Nvcc:
#if PROTEUS_ENABLE_CUDA
    return createCppJitCompilerNvcc();
#else
    reportFatalError("NVCC backend is not available in this build");
#endif
  }

  reportFatalError("Unsupported CppJit compiler backend");
}

} // namespace proteus
