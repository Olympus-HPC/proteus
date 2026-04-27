#include "proteus/impl/Frontend/CppJitCompiler.h"
#include "proteus/TimeTracing.h"
#include "proteus/impl/Config.h"

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/Program.h>

#include <mutex>
#include <optional>

namespace proteus {

namespace {

std::string resolveProgramPath(llvm::StringRef Program) {
  if (llvm::sys::path::has_parent_path(Program)) {
    if (!llvm::sys::fs::can_execute(Program))
      return {};

    llvm::SmallString<256> RealPath;
    if (!llvm::sys::fs::real_path(Program, RealPath))
      return RealPath.str().str();
    return Program.str();
  }

  auto Found = llvm::sys::findProgramByName(Program);
  if (!Found)
    return {};
  return *Found;
}

ResolvedToolPath resolveToolPath(
    llvm::StringRef Program, const std::optional<const std::string> &Override,
    std::optional<llvm::StringRef> ToolchainHintDir = std::nullopt) {
  if (Override) {
    return {resolveProgramPath(*Override), *Override};
  }

  if (ToolchainHintDir) {
    llvm::SmallString<256> Candidate(*ToolchainHintDir);
    llvm::sys::path::append(Candidate, Program);
    std::string Path = resolveProgramPath(Candidate);
    if (!Path.empty()) {
      return {std::move(Path),
              "LLVM toolchain hint " + ToolchainHintDir->str()};
    }
  }

  return {resolveProgramPath(Program), "PATH"};
}

const ResolvedToolPath &
resolveTool(std::mutex &CacheMutex, std::optional<ResolvedToolPath> &Cache,
            llvm::StringRef Program,
            const std::optional<const std::string> &Override,
            llvm::StringRef OverrideVar, llvm::StringRef MissingContext,
            std::optional<llvm::StringRef> ToolchainHintDir = std::nullopt) {
  // Fast path if it is already cached.
  if (Cache)
    return *Cache;

  std::lock_guard<std::mutex> Lock(CacheMutex);

  // Another thread may have populated the cache while we were waiting.
  if (Cache)
    return *Cache;

  ResolvedToolPath Tool = resolveToolPath(Program, Override, ToolchainHintDir);

  if (Tool.Path.empty()) {
    if (Override) {
      reportFatalError("Failed to resolve " + Program.str() + " from " +
                       OverrideVar.str() + "=" + *Override);
    }
    reportFatalError(MissingContext.str() + " Set " + OverrideVar.str() +
                     " or ensure " + Program.str() + " is on PATH.");
  }

  if (Override)
    Tool.Origin = OverrideVar.str() + "=" + Tool.Origin;

  Cache = std::move(Tool);
  return *Cache;
}

} // namespace

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
  if (Backend == CppJitCompilerBackend::Clang)
    H = hashCombine(H, hashCodeGenConfig(Config::get().getCGConfig()));
  for (const auto &Arg : ExtraArgs)
    H = hashCombine(H, hash(Arg));
  return H;
}

std::unique_ptr<CppJitCompiler>
createCppJitCompiler(const CppJitCompileRequest &Request) {
  TIMESCOPE("proteus::createCppJitCompiler");
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

const ResolvedToolPath &resolveClangxx() {
  static std::mutex CacheMutex;
  static std::optional<ResolvedToolPath> Cache;

#ifdef PROTEUS_LLVM_TOOLS_BINDIR
  constexpr llvm::StringRef ToolchainHintDir = PROTEUS_LLVM_TOOLS_BINDIR;
#else
  constexpr std::optional<llvm::StringRef> ToolchainHintDir = std::nullopt;
#endif

  return resolveTool(CacheMutex, Cache, "clang++",
                     Config::get().ProteusClangxxBin, "PROTEUS_CLANGXX_BIN",
                     "Failed to resolve required host compiler clang++.",
                     ToolchainHintDir);
}

#if PROTEUS_ENABLE_CUDA
const ResolvedToolPath &resolveNvcc() {
  static std::mutex CacheMutex;
  static std::optional<ResolvedToolPath> Cache;
  return resolveTool(CacheMutex, Cache, "nvcc", Config::get().ProteusNvccBin,
                     "PROTEUS_NVCC_BIN",
                     "CUDA support is enabled in this build, but nvcc was not "
                     "found.");
}
#endif

} // namespace proteus
