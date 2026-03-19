#include "proteus/impl/Frontend/CppJitCompiler.h"
#include "proteus/TimeTracing.h"

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
#if LLVM_VERSION_MAJOR >= 20
  Compiler.createDiagnostics(*vfs::getRealFileSystem());
#else
  Compiler.createDiagnostics();
#endif
}

// Clang-backed implementation for IR and shared-library compilation paths.
class CppJitCompilerClang : public CppJitCompiler {
private:
  CppJitArtifact compileToDynamicLibrary(const CppJitCompileRequest &Request) {
    TIMESCOPE(CppJitCompilerClang, compileToDynamicLibrary);
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
        PROTEUS_CLANGXX_BIN,
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

    clang::driver::Driver D(PROTEUS_CLANGXX_BIN, sys::getDefaultTargetTriple(),
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
    CompilerInstance Compiler;
    initializeCompilerInstance(Compiler);

    std::string SourceName = Request.ModuleHash.toString() + ".cpp";

    std::vector<std::string> ArgStorage;
    if (Request.TargetModel == TargetModelType::HOST) {
      ArgStorage = {PROTEUS_CLANGXX_BIN,
                    "-emit-llvm",
                    "-S",
                    "-std=c++17",
                    CppJitCompiler::FrontendOptLevelFlag,
                    "-x",
                    "c++",
                    "-fPIC",
                    SourceName};
    } else {
      std::string OffloadArch = "--offload-arch=" + Request.DeviceArch;
      ArgStorage = {
          PROTEUS_CLANGXX_BIN,
          "-emit-llvm",
          "-S",
          "-std=c++17",
          CppJitCompiler::FrontendOptLevelFlag,
          "-x",
          (Request.TargetModel == TargetModelType::HIP ? "hip" : "cuda"),
          "--offload-device-only",
          OffloadArch,
          SourceName};
    }

    ArgStorage.insert(ArgStorage.end(), Request.ExtraArgs.begin(),
                      Request.ExtraArgs.end());

    std::vector<const char *> DriverArgs;
    DriverArgs.reserve(ArgStorage.size());
    for (const auto &S : ArgStorage)
      DriverArgs.push_back(S.c_str());

    clang::driver::Driver D(PROTEUS_CLANGXX_BIN, sys::getDefaultTargetTriple(),
                            Compiler.getDiagnostics());
    D.setCheckInputsExist(false);
    auto *C = D.BuildCompilation(DriverArgs);
    if (!C || Compiler.getDiagnostics().hasErrorOccurred())
      reportFatalError("Building Driver failed");

    const clang::driver::JobList &Jobs = C->getJobs();
    if (Jobs.empty())
      reportFatalError("Expected compilation job, found empty joblist");

    const clang::driver::Command &Cmd =
        llvm::cast<clang::driver::Command>(*Jobs.begin());
    const auto &CC1Args = Cmd.getArguments();
    if (!llvm::is_contained(CC1Args, StringRef{"-cc1"}))
      reportFatalError("Expected first job to be the compilation");

    auto Invocation = std::make_shared<CompilerInvocation>();
    if (!CompilerInvocation::CreateFromArgs(*Invocation, CC1Args,
                                            Compiler.getDiagnostics())) {
      throw std::runtime_error("Failed to create compiler invocation");
    }

    Compiler.setInvocation(Invocation);
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
