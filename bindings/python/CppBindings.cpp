#include "PythonBindings.h"

#include "proteus/CppJitModule.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace proteus_python {
namespace {

// Keep the Python-facing compiler strings aligned with the C++ JIT backends.
proteus::CppJitCompilerBackend parseCompiler(const std::string &Compiler) {
  if (Compiler == "clang")
    return proteus::CppJitCompilerBackend::Clang;
  if (Compiler == "nvcc")
    return proteus::CppJitCompilerBackend::Nvcc;
  throw py::value_error("compiler must be 'clang' or 'nvcc'");
}

class CppModule final : public ModuleBase {
  // Owns the concrete C++ JIT implementation behind the generic Python ABI.
  proteus::CppJitModule M;

public:
  CppModule(const std::string &Target, const std::string &Code,
            const std::vector<std::string> &ExtraArgs,
            proteus::CppJitCompilerBackend Compiler)
      : M(Target, Code, ExtraArgs, Compiler) {}

  // The Python bindings call through ModuleBase so both C++ and MLIR modules
  // can share the same wrapper types.
  void compile(bool) override { M.compile(); }
  void *getFunctionAddress(const std::string &Name) override {
    return M.getFunctionAddress(Name);
  }
  proteus::TargetModelType getTargetModel() const override {
    return M.getTargetModel();
  }
  void *getKernelAddress(const std::string &Name) override {
    // The concrete module enforces that kernels only come from device targets.
    return M.getKernelAddress(Name);
  }
  proteus::DispatchResult launch(void *KernelFunc, LaunchDims GridDim,
                                 LaunchDims BlockDim, void *KernelArgs[],
                                 uint64_t ShmemSize, void *Stream) override {
    return M.launch(KernelFunc, GridDim, BlockDim, KernelArgs, ShmemSize,
                    Stream);
  }
};

} // namespace

std::shared_ptr<ModuleBase>
createCppModule(const std::string &Target, const std::string &Code,
                const std::vector<std::string> &ExtraArgs,
                const std::string &Compiler) {
  return std::make_shared<CppModule>(Target, Code, ExtraArgs,
                                     parseCompiler(Compiler));
}

} // namespace proteus_python
