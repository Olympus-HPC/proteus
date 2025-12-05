#include <cstdint>
#include <memory>

#include <clang/CodeGen/CodeGenAction.h>
#include <clang/Driver/Compilation.h>
#include <clang/Driver/Driver.h>
#include <clang/Driver/Tool.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/CompilerInvocation.h>
#include <clang/Lex/PreprocessorOptions.h>
#include <llvm/IR/Module.h>
#include <llvm/Option/ArgList.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/TargetParser/Triple.h>

#include "proteus/CppJitModule.hpp"
#include "proteus/Hashing.hpp"

namespace proteus {

#if PROTEUS_ENABLE_CUDA
static std::vector<std::string> ExtraToolchainArgs = {"-L" PROTEUS_CUDA_LIBDIR,
                                                      "-lcudart"};
#else
static std::vector<std::string> ExtraToolchainArgs = {};
#endif

using namespace clang;
using namespace llvm;

CppJitModule::CppJitModule(TargetModelType TargetModel, StringRef Code,
                           const std::vector<std::string> &ExtraArgs)
    : TargetModel(TargetModel), Code(Code.str()), ModuleHash(hash(Code)),
      ExtraArgs(ExtraArgs), Dispatch(Dispatcher::getDispatcher(TargetModel)) {}
CppJitModule::CppJitModule(StringRef Target, StringRef Code,
                           const std::vector<std::string> &ExtraArgs)
    : TargetModel(parseTargetModel(Target)), Code(Code), ModuleHash(hash(Code)),
      ExtraArgs(ExtraArgs), Dispatch(Dispatcher::getDispatcher(TargetModel)) {}

void CppJitModule::compileCppToDynamicLibrary() {
  // Create compiler instance.
  CompilerInstance Compiler;
  // Create diagnostics engine.
#if LLVM_VERSION_MAJOR >= 20
  Compiler.createDiagnostics(*vfs::getRealFileSystem());
#else
  Compiler.createDiagnostics();
#endif

  // We build a driver to compile to a dynamic library object.
  // Create temp source file
  SmallString<128> SourcePath;
  std::error_code EC =
      sys::fs::createTemporaryFile("proteus", "cpp", SourcePath);
  if (EC)
    PROTEUS_FATAL_ERROR("Failed to create temp source file");

  // Write source code to temp file
  {
    raw_fd_ostream OS(SourcePath, EC);
    if (EC)
      PROTEUS_FATAL_ERROR("Failed to write source file");
    OS << Code;
  }

  // Create temp output file path
  SmallString<128> OutputPath;
  EC = sys::fs::createTemporaryFile("proteus", "so", OutputPath);
  if (EC)
    PROTEUS_FATAL_ERROR("Failed to create temp output file");

  std::string OffloadArch = "--offload-arch=" + Dispatch.getDeviceArch().str();

  std::vector<std::string> ArgStorage = {
      PROTEUS_CLANGXX_BIN,
      "-shared",
      "-std=c++17",
      FrontendOptLevelFlag,
      "-x",
      (TargetModel == TargetModelType::HOST_HIP ? "hip" : "cuda"),
      "-fPIC",
      OffloadArch,
      "-o",
      OutputPath.c_str(),
      SourcePath.c_str()};

  ArgStorage.insert(ArgStorage.end(), ExtraToolchainArgs.begin(),
                    ExtraToolchainArgs.end());
  ArgStorage.insert(ArgStorage.end(), ExtraArgs.begin(), ExtraArgs.end());

  std::vector<const char *> DriverArgs;
  for (const auto &S : ArgStorage)
    DriverArgs.push_back(S.c_str());

  // Create driver
  clang::driver::Driver D(PROTEUS_CLANGXX_BIN, sys::getDefaultTargetTriple(),
                          Compiler.getDiagnostics());

  auto *C = D.BuildCompilation(DriverArgs);
  if (!C || Compiler.getDiagnostics().hasErrorOccurred())
    PROTEUS_FATAL_ERROR("Building Driver failed");

  // Extract the argument from the compilation job.
  const clang::driver::JobList &Jobs = C->getJobs();
  if (Jobs.empty())
    PROTEUS_FATAL_ERROR("Expected compilation job, found empty joblist");

  // Execute ALL jobs (device compilation, bundling, host compilation)
  SmallVector<std::pair<int, const clang::driver::Command *>, 4>
      FailingCommands;
  int Res = D.ExecuteCompilation(*C, FailingCommands);

  if (Res != 0 || !FailingCommands.empty()) {
    sys::fs::remove(SourcePath);
    sys::fs::remove(OutputPath);
    PROTEUS_FATAL_ERROR("Compilation failed");
  }

  // Register the dynamic library file with the dispatcher to make it available
  // post compilation.
  Dispatch.registerDynamicLibrary(ModuleHash, OutputPath);

  sys::fs::remove(SourcePath);
  sys::fs::remove(OutputPath);
}

CppJitModule::CompilationResult CppJitModule::compileCppToIR() {
  // Create compiler instance.
  CompilerInstance Compiler;
  // Create diagnostics engine.
#if LLVM_VERSION_MAJOR >= 20
  Compiler.createDiagnostics(*vfs::getRealFileSystem());
#else
  Compiler.createDiagnostics();
#endif

  // Hashing should treat Code as a pointer and size to hash all bytes.
  std::string SourceName = ModuleHash.toString() + ".cpp";

  // Set up driver arguments and build a driver invocation to extract cc1
  // arguments. This is needed to retrieve system include paths that the driver
  // discovers and other options that are needed for source lowering.
  std::vector<std::string> ArgStorage;
  if (TargetModel == TargetModelType::HOST) {
    ArgStorage = {PROTEUS_CLANGXX_BIN,
                  "-emit-llvm",
                  "-S",
                  "-std=c++17",
                  FrontendOptLevelFlag,
                  "-x",
                  "c++",
                  "-fPIC",
                  SourceName};
  } else {
    std::string OffloadArch =
        "--offload-arch=" + Dispatch.getDeviceArch().str();
    ArgStorage = {PROTEUS_CLANGXX_BIN,
                  "-emit-llvm",
                  "-S",
                  "-std=c++17",
                  FrontendOptLevelFlag,
                  "-x",
                  (TargetModel == TargetModelType::HIP ? "hip" : "cuda"),
                  "--offload-device-only",
                  OffloadArch,
                  SourceName};
  }

  ArgStorage.insert(ArgStorage.end(), ExtraArgs.begin(), ExtraArgs.end());

  std::vector<const char *> DriverArgs;
  DriverArgs.reserve(ArgStorage.size());
  for (const auto &S : ArgStorage)
    DriverArgs.push_back(S.c_str());

  clang::driver::Driver D(PROTEUS_CLANGXX_BIN, sys::getDefaultTargetTriple(),
                          Compiler.getDiagnostics());
  // We compile in-memory, make file checking false.
  D.setCheckInputsExist(false);
  auto *C = D.BuildCompilation(DriverArgs);
  if (!C || Compiler.getDiagnostics().hasErrorOccurred())
    PROTEUS_FATAL_ERROR("Building Driver failed");

  // Extract the argument from the compilation job.
  const clang::driver::JobList &Jobs = C->getJobs();
  if (Jobs.empty())
    PROTEUS_FATAL_ERROR("Expected compilation job, found empty joblist");

  const clang::driver::Command &Cmd =
      llvm::cast<clang::driver::Command>(*Jobs.begin());
  const auto &CC1Args = Cmd.getArguments();
  // Use StringRef to make sure we compare by value and pointers.
  if (!llvm::is_contained(CC1Args, StringRef{"-cc1"}))
    PROTEUS_FATAL_ERROR("Expected first job to be the compilation");

  // Create compiler invocation with minimal arguments.
  auto Invocation = std::make_shared<CompilerInvocation>();
  // Create invocation from arguments
  if (!CompilerInvocation::CreateFromArgs(*Invocation,
                                          CC1Args, // Skip program name
                                          Compiler.getDiagnostics())) {
    throw std::runtime_error("Failed to create compiler invocation");
  }

  // Set the invocation.
  Compiler.setInvocation(Invocation);

  // Load clang plugins, if there are any.
  Compiler.LoadRequestedPlugins();

  std::unique_ptr<MemoryBuffer> Buffer =
      MemoryBuffer::getMemBuffer(Code, SourceName);

  // Add the buffer to preprocessor options and transfer ownership.
  Compiler.getPreprocessorOpts().addRemappedFile(SourceName, Buffer.release());

  // Create the LLVM IR generating action.
  EmitLLVMOnlyAction Action;

  if (!Compiler.ExecuteAction(Action))
    PROTEUS_FATAL_ERROR("Failed to execute action");

  std::unique_ptr<llvm::Module> Module = Action.takeModule();
  if (!Module)
    PROTEUS_FATAL_ERROR("Failed to take LLVM module");

  std::unique_ptr<LLVMContext> Ctx{Action.takeLLVMContext()};

  return CppJitModule::CompilationResult{std::move(Ctx), std::move(Module)};
}

void CppJitModule::compile() {
  // Lookup in the object cache of the dispatcher before lowering the cpp code
  // to LLVM IR.
  if ((Library = Dispatch.lookupCompiledLibrary(ModuleHash))) {
    IsCompiled = true;
    return;
  }

  switch (TargetModel) {
  case TargetModelType::HOST_HIP:
  case TargetModelType::HOST_CUDA:
    compileCppToDynamicLibrary();
    // Retrieve the compiled library from the dispatcher which stores the
    // dynamic library file.
    Library = Dispatch.lookupCompiledLibrary(ModuleHash);
    if (!Library)
      PROTEUS_FATAL_ERROR("Expected non-null library after compilation");
    break;
  default:
    auto CRes = compileCppToIR();
    auto ObjectModule =
        Dispatch.compile(std::move(CRes.Ctx), std::move(CRes.Mod), ModuleHash,
                         /*DisableIROpt=*/true);
    Library = std::make_unique<CompiledLibrary>(std::move(ObjectModule));
  }

  IsCompiled = true;
}

} // namespace proteus
