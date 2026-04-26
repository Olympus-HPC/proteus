#include "proteus/TimeTracing.h"
#include "proteus/impl/Frontend/CppJitCompiler.h"

#if PROTEUS_ENABLE_CUDA

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Program.h>
#include <llvm/Support/raw_ostream.h>

#include <optional>

namespace proteus {

using namespace llvm;

namespace {

std::string readTempFile(const SmallString<128> &Path) {
  auto Buf = MemoryBuffer::getFileAsStream(Path.str());
  if (!Buf)
    return std::string{};
  return (*Buf)->getBuffer().str();
}

// NVCC-backed implementation for cubin and HOST_CUDA launcher builds.
class CppJitCompilerNvcc : public CppJitCompiler {
private:
  CppJitArtifact compileToCubin(const CppJitCompileRequest &Request) {
    TIMESCOPE(CppJitCompilerNvcc, compileToCubin);
    const ResolvedToolPath &Nvcc = resolveNvcc();
    SmallString<128> SourcePath;
    std::error_code EC =
        sys::fs::createTemporaryFile("proteus", "cu", SourcePath);
    if (EC)
      reportFatalError("Failed to create temp nvcc source file");

    SmallString<128> OutputPath;
    EC = sys::fs::createTemporaryFile("proteus", "cubin", OutputPath);
    if (EC)
      reportFatalError("Failed to create temp nvcc output file");

    SmallString<128> StdoutPath;
    EC = sys::fs::createTemporaryFile("proteus", "nvcc.stdout", StdoutPath);
    if (EC)
      reportFatalError("Failed to create temp nvcc stdout file");

    SmallString<128> StderrPath;
    EC = sys::fs::createTemporaryFile("proteus", "nvcc.stderr", StderrPath);
    if (EC)
      reportFatalError("Failed to create temp nvcc stderr file");

    {
      raw_fd_ostream OS(SourcePath, EC);
      if (EC)
        reportFatalError("Failed to write nvcc source file");
      OS << Request.Code;
    }

    std::vector<std::string> ArgStorage = {Nvcc.Path,
                                           "-std=c++17",
                                           "-cubin",
                                           "--gpu-architecture=" +
                                               Request.DeviceArch,
                                           CppJitCompiler::FrontendOptLevelFlag,
                                           "-o",
                                           OutputPath.c_str(),
                                           SourcePath.c_str()};

    ArgStorage.insert(ArgStorage.end(), Request.ExtraArgs.begin(),
                      Request.ExtraArgs.end());

    std::vector<StringRef> Args;
    Args.reserve(ArgStorage.size());
    for (const auto &Arg : ArgStorage)
      Args.push_back(Arg);

    std::optional<StringRef> Redirects[] = {std::nullopt,
                                            StringRef(StdoutPath.c_str()),
                                            StringRef(StderrPath.c_str())};

    std::string ErrMsg;
    bool ExecutionFailed = false;
    int Res = sys::ExecuteAndWait(Nvcc.Path, Args, std::nullopt, Redirects, 0,
                                  0, &ErrMsg, &ExecutionFailed);

    if (ExecutionFailed || Res != 0) {
      std::string Command;
      for (const auto &Arg : ArgStorage) {
        if (!Command.empty())
          Command += ' ';
        Command += Arg;
      }

      std::string Diag;
      if (!ErrMsg.empty())
        Diag += "Process error: " + ErrMsg + "\n";

      std::string Stdout = readTempFile(StdoutPath);
      if (!Stdout.empty())
        Diag += "stdout:\n" + Stdout + "\n";

      std::string Stderr = readTempFile(StderrPath);
      if (!Stderr.empty())
        Diag += "stderr:\n" + Stderr + "\n";

      reportFatalError("NVCC compilation failed with exit code " +
                       std::to_string(Res) + "\nCommand: " + Command +
                       (Diag.empty() ? std::string{} : "\n" + Diag));
    }

    auto Cubin = MemoryBuffer::getFileAsStream(OutputPath.str());
    if (!Cubin)
      reportFatalError("Failed to read nvcc cubin output");

    sys::fs::remove(SourcePath);
    sys::fs::remove(OutputPath);
    sys::fs::remove(StdoutPath);
    sys::fs::remove(StderrPath);

    return CppJitArtifact::deviceBinary(MemoryBuffer::getMemBufferCopy(
        (*Cubin)->getBuffer(), "proteus-nvcc-cubin"));
  }

  CppJitArtifact compileToDynamicLibrary(const CppJitCompileRequest &Request) {
    TIMESCOPE(CppJitCompilerNvcc, compileToDynamicLibrary);
    const ResolvedToolPath &Nvcc = resolveNvcc();
    SmallString<128> SourcePath;
    std::error_code EC =
        sys::fs::createTemporaryFile("proteus", "cu", SourcePath);
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

    SmallString<128> StdoutPath;
    EC = sys::fs::createTemporaryFile("proteus", "nvcc.stdout", StdoutPath);
    if (EC)
      reportFatalError("Failed to create temp nvcc stdout file");

    SmallString<128> StderrPath;
    EC = sys::fs::createTemporaryFile("proteus", "nvcc.stderr", StderrPath);
    if (EC)
      reportFatalError("Failed to create temp nvcc stderr file");

    std::vector<std::string> ArgStorage = {Nvcc.Path,
                                           "-shared",
                                           "-std=c++17",
                                           CppJitCompiler::FrontendOptLevelFlag,
                                           "--compiler-options=-fPIC",
                                           "--gpu-architecture=" +
                                               Request.DeviceArch,
                                           "-o",
                                           OutputPath.c_str(),
                                           SourcePath.c_str()};

    ArgStorage.insert(ArgStorage.end(), Request.ExtraArgs.begin(),
                      Request.ExtraArgs.end());

    std::vector<StringRef> Args;
    Args.reserve(ArgStorage.size());
    for (const auto &Arg : ArgStorage)
      Args.push_back(Arg);

    std::optional<StringRef> Redirects[] = {std::nullopt,
                                            StringRef(StdoutPath.c_str()),
                                            StringRef(StderrPath.c_str())};

    std::string ErrMsg;
    bool ExecutionFailed = false;
    int Res = sys::ExecuteAndWait(Nvcc.Path, Args, std::nullopt, Redirects, 0,
                                  0, &ErrMsg, &ExecutionFailed);

    if (ExecutionFailed || Res != 0) {
      std::string Command;
      for (const auto &Arg : ArgStorage) {
        if (!Command.empty())
          Command += ' ';
        Command += Arg;
      }

      std::string Diag;
      if (!ErrMsg.empty())
        Diag += "Process error: " + ErrMsg + "\n";

      std::string Stdout = readTempFile(StdoutPath);
      if (!Stdout.empty())
        Diag += "stdout:\n" + Stdout + "\n";

      std::string Stderr = readTempFile(StderrPath);
      if (!Stderr.empty())
        Diag += "stderr:\n" + Stderr + "\n";

      reportFatalError(
          "NVCC shared library compilation failed with exit code " +
          std::to_string(Res) + "\nCommand: " + Command +
          (Diag.empty() ? std::string{} : "\n" + Diag));
    }

    sys::fs::remove(SourcePath);
    sys::fs::remove(StdoutPath);
    sys::fs::remove(StderrPath);

    return CppJitArtifact::sharedLibrary(OutputPath.c_str());
  }

public:
  CppJitArtifact compile(const CppJitCompileRequest &Request) override {
    TIMESCOPE(CppJitCompilerNvcc, compile);
    switch (Request.TargetModel) {
    case TargetModelType::CUDA:
      return compileToCubin(Request);
    case TargetModelType::HOST_CUDA:
      return compileToDynamicLibrary(Request);
    default:
      reportFatalError(
          "NVCC backend is supported only for CUDA and HOST_CUDA targets");
    }
  }
};

} // namespace

std::unique_ptr<CppJitCompiler> createCppJitCompilerNvcc() {
  return std::make_unique<CppJitCompilerNvcc>();
}

} // namespace proteus

#endif
