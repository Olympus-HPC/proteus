#ifndef PROTEUS_RUNTIME_FRONTEND_CPPJITCOMPILER_H
#define PROTEUS_RUNTIME_FRONTEND_CPPJITCOMPILER_H

#include "proteus/CppJitCompilerBackend.h"
#include "proteus/Frontend/TargetModel.h"
#include "proteus/impl/Hashing.h"

#include <memory>
#include <string>
#include <vector>

namespace llvm {
class LLVMContext;
class MemoryBuffer;
class Module;
} // namespace llvm

namespace proteus {

struct ResolvedToolPath {
  std::string Path;
  std::string Origin;
};

// Compiler-facing input assembled by CppJitModule.
struct CppJitCompileRequest {
  TargetModelType TargetModel;
  CppJitCompilerBackend Backend;
  const std::string &Code;
  const std::vector<std::string> &ExtraArgs;
  const HashT &ModuleHash;
  std::string DeviceArch;
};

// Transport object returned by the compiler layer before registration.
struct CppJitArtifact {
  enum class Kind { SharedLibrary, DeviceBinary, LLVMIR };

  Kind ArtifactKind;
  std::string Path;
  std::unique_ptr<llvm::MemoryBuffer> ObjectBuffer;
  std::unique_ptr<llvm::LLVMContext> Ctx;
  std::unique_ptr<llvm::Module> Mod;

  CppJitArtifact(CppJitArtifact &&);
  CppJitArtifact &operator=(CppJitArtifact &&);
  ~CppJitArtifact();

  static CppJitArtifact sharedLibrary(std::string Path);
  static CppJitArtifact deviceBinary(std::unique_ptr<llvm::MemoryBuffer> Obj);
  static CppJitArtifact llvmIR(std::unique_ptr<llvm::LLVMContext> Ctx,
                               std::unique_ptr<llvm::Module> Mod);

private:
  CppJitArtifact();
};

// Abstract compiler interface used by CppJitModule.
class CppJitCompiler {
public:
  static constexpr const char *FrontendOptLevelFlag = "-O3";

  static bool isBackendSupported(TargetModelType TM,
                                 CppJitCompilerBackend Backend);
  virtual ~CppJitCompiler() = default;
  virtual CppJitArtifact compile(const CppJitCompileRequest &Request) = 0;
};

HashT computeCppJitModuleHash(TargetModelType TM, CppJitCompilerBackend Backend,
                              const std::string &Code,
                              const std::vector<std::string> &ExtraArgs);
const ResolvedToolPath &resolveClangxx();
#if PROTEUS_ENABLE_CUDA
const ResolvedToolPath &resolveNvcc();
#endif
std::unique_ptr<CppJitCompiler>
createCppJitCompiler(const CppJitCompileRequest &Request);

std::unique_ptr<CppJitCompiler> createCppJitCompilerClang();
#if PROTEUS_ENABLE_CUDA
std::unique_ptr<CppJitCompiler> createCppJitCompilerNvcc();
#endif

} // namespace proteus

#endif
