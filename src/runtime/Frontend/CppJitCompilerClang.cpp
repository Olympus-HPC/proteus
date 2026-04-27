#include "proteus/TimeTracing.h"
#include "proteus/impl/Frontend/CppJitCompiler.h"

#include <clang/CodeGen/CodeGenAction.h>
#include <clang/Driver/Compilation.h>
#include <clang/Driver/Driver.h>
#include <clang/Driver/Tool.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/CompilerInvocation.h>
#include <clang/Lex/PreprocessorOptions.h>

#include <llvm/ADT/STLExtras.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/TargetParser/Triple.h>

#include <memory>
namespace proteus {

#if PROTEUS_ENABLE_CUDA
static std::vector<std::string> ExtraToolchainArgs = {"-L" PROTEUS_CUDA_LIBDIR,
                                                      "-lcudart"};
#else
static std::vector<std::string> ExtraToolchainArgs = {};
#endif

using namespace clang;
using namespace llvm;

namespace {

void initializeCompilerInstance(CompilerInstance &Compiler) {
#if LLVM_VERSION_MAJOR >= 22
  Compiler.createDiagnostics();
#elif LLVM_VERSION_MAJOR >= 20
  Compiler.createDiagnostics(*vfs::getRealFileSystem());
#else
  Compiler.createDiagnostics();
#endif
  if (!Compiler.hasDiagnostics())
    reportFatalError("Compiler instance has no diagnostics");
}

std::shared_ptr<CompilerInvocation> createCompilerInvocation(
    const ResolvedToolPath &Clangxx, const std::vector<std::string> &ArgStorage,
    CompilerInstance &Compiler, llvm::StringRef ExpectedContext,
    bool CheckInputsExist = false) {
  std::vector<const char *> DriverArgs;
  DriverArgs.reserve(ArgStorage.size());
  for (const auto &S : ArgStorage)
    DriverArgs.push_back(S.c_str());

  clang::driver::Driver D(Clangxx.Path, sys::getDefaultTargetTriple(),
                          Compiler.getDiagnostics());
  D.setCheckInputsExist(CheckInputsExist);

  auto C = std::unique_ptr<clang::driver::Compilation>(
      D.BuildCompilation(DriverArgs));
  if (!C || Compiler.getDiagnostics().hasErrorOccurred()) {
    reportFatalError(ExpectedContext.str() +
                     ": failed to build Clang compilation with compiler '" +
                     Clangxx.Path + "' selected from " + Clangxx.Origin + ".");
  }

  if (C->getJobs().empty())
    reportFatalError(ExpectedContext.str() +
                     ": expected compilation job, found empty joblist");

  const clang::driver::Command &Cmd =
      llvm::cast<clang::driver::Command>(*C->getJobs().begin());
  const auto &CC1Args = Cmd.getArguments();
  if (!llvm::is_contained(CC1Args, StringRef{"-cc1"})) {
    reportFatalError(ExpectedContext.str() +
                     ": expected first job to be the compilation");
  }

  auto Invocation = std::make_shared<CompilerInvocation>();
  if (!CompilerInvocation::CreateFromArgs(*Invocation, CC1Args,
                                          Compiler.getDiagnostics())) {
    reportFatalError(ExpectedContext.str() +
                     ": failed to create compiler invocation");
  }

  return Invocation;
}

// Clang-backed implementation for IR and shared-library compilation paths.
class CppJitCompilerClang : public CppJitCompiler {
private:
  CppJitArtifact compileToDynamicLibrary(const CppJitCompileRequest &Request) {
    TIMESCOPE(CppJitCompilerClang, compileToDynamicLibrary);
    const ResolvedToolPath &Clangxx = resolveClangxx();
    CompilerInstance Compiler;
    initializeCompilerInstance(Compiler);

    SmallString<128> SourcePath;
    std::error_code EC =
        sys::fs::createTemporaryFile("proteus", "cpp", SourcePath);
    if (EC)
      reportFatalError("Failed to create temp source file");

    {
      raw_fd_ostream OS(SourcePath, EC);
      if (EC)
        reportFatalError("Failed to write source file");
      OS << Request.Code;
    }

    SmallString<128> OutputPath;
    EC = sys::fs::createTemporaryFile("proteus", "so", OutputPath);
    if (EC)
      reportFatalError("Failed to create temp output file");

    std::string OffloadArch = "--offload-arch=" + Request.DeviceArch;

    std::vector<std::string> ArgStorage = {
        Clangxx.Path,
        "-shared",
        "-std=c++17",
        CppJitCompiler::FrontendOptLevelFlag,
        "-x",
        (Request.TargetModel == TargetModelType::HOST_HIP ? "hip" : "cuda"),
        "-fPIC",
        OffloadArch,
        "-o",
        OutputPath.c_str(),
        SourcePath.c_str()};

    ArgStorage.insert(ArgStorage.end(), ExtraToolchainArgs.begin(),
                      ExtraToolchainArgs.end());
    ArgStorage.insert(ArgStorage.end(), Request.ExtraArgs.begin(),
                      Request.ExtraArgs.end());

    std::vector<const char *> DriverArgs;
    for (const auto &S : ArgStorage)
      DriverArgs.push_back(S.c_str());

    clang::driver::Driver D(Clangxx.Path, sys::getDefaultTargetTriple(),
                            Compiler.getDiagnostics());

    auto *C = D.BuildCompilation(DriverArgs);
    if (!C || Compiler.getDiagnostics().hasErrorOccurred())
      reportFatalError("Building Driver failed");

    const clang::driver::JobList &Jobs = C->getJobs();
    if (Jobs.empty())
      reportFatalError("Expected compilation job, found empty joblist");

    SmallVector<std::pair<int, const clang::driver::Command *>, 4>
        FailingCommands;
    int Res = D.ExecuteCompilation(*C, FailingCommands);

    if (Res != 0 || !FailingCommands.empty())
      reportFatalError("Compilation failed");

    sys::fs::remove(SourcePath);
    return CppJitArtifact::sharedLibrary(OutputPath.c_str());
  }

  CppJitArtifact compileToIR(const CppJitCompileRequest &Request) {
    TIMESCOPE(CppJitCompilerClang, compileToIR);
    const ResolvedToolPath &Clangxx = resolveClangxx();
    CompilerInstance TempCompiler;
    initializeCompilerInstance(TempCompiler);

    std::string SourceName = Request.ModuleHash.toString() + ".cpp";

    std::vector<std::string> ArgStorage;
    // Keep Clang in optimized frontend/codegen mode so it emits optimized-mode
    // IR features such as fmuladd, but skip Clang's LLVM pass pipeline. Proteus
    // runs its configured middle-end pipeline when the dispatcher compiles the
    // returned LLVM IR.
    if (Request.TargetModel == TargetModelType::HOST) {
      ArgStorage = {Clangxx.Path,
                    "-emit-llvm",
                    "-S",
                    "-std=c++17",
                    CppJitCompiler::FrontendOptLevelFlag,
                    "-Xclang",
                    "-disable-llvm-passes",
                    "-x",
                    "c++",
                    "-fPIC",
                    SourceName};
    } else {
      std::string OffloadArch = "--offload-arch=" + Request.DeviceArch;
      ArgStorage = {
          Clangxx.Path,
          "-emit-llvm",
          "-S",
          "-std=c++17",
          CppJitCompiler::FrontendOptLevelFlag,
          "-Xclang",
          "-disable-llvm-passes",
          "-x",
          (Request.TargetModel == TargetModelType::HIP ? "hip" : "cuda"),
          "--offload-device-only",
          OffloadArch,
          SourceName};
    }

    ArgStorage.insert(ArgStorage.end(), Request.ExtraArgs.begin(),
                      Request.ExtraArgs.end());

    auto Invocation = createCompilerInvocation(Clangxx, ArgStorage,
                                               TempCompiler, "Clang IR build");

#if LLVM_VERSION_MAJOR >= 22
    CompilerInstance Compiler(Invocation);
#else
    CompilerInstance Compiler;
    Compiler.setInvocation(Invocation);
#endif
    initializeCompilerInstance(Compiler);
    Compiler.LoadRequestedPlugins();

    std::unique_ptr<MemoryBuffer> Buffer =
        MemoryBuffer::getMemBuffer(Request.Code, SourceName);
    Compiler.getPreprocessorOpts().addRemappedFile(SourceName,
                                                   Buffer.release());

    EmitLLVMOnlyAction Action;
    if (!Compiler.ExecuteAction(Action))
      reportFatalError("Failed to execute action");

    std::unique_ptr<llvm::Module> Module = Action.takeModule();
    if (!Module)
      reportFatalError("Failed to take LLVM module");

    std::unique_ptr<LLVMContext> Ctx{Action.takeLLVMContext()};
    return CppJitArtifact::llvmIR(std::move(Ctx), std::move(Module));
  }

public:
  CppJitArtifact compile(const CppJitCompileRequest &Request) override {
    TIMESCOPE(CppJitCompilerClang, compile);
    switch (Request.TargetModel) {
    case TargetModelType::HOST_CUDA:
    case TargetModelType::HOST_HIP:
      return compileToDynamicLibrary(Request);
    default:
      return compileToIR(Request);
    }
  }
};

} // namespace

std::unique_ptr<CppJitCompiler> createCppJitCompilerClang() {
  return std::make_unique<CppJitCompilerClang>();
}

} // namespace proteus
