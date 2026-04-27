#include "PythonBindings.h"

#include "proteus/MLIRJitModule.h"

namespace proteus_python {
namespace {

class MLIRModule final : public ModuleBase {
  // Owns the concrete MLIR JIT implementation behind the shared Python ABI.
  proteus::MLIRJitModule M;

public:
  MLIRModule(const std::string &Target, const std::string &Code)
      : M(Target, Code) {}

  // Route the generic binding entrypoints to the MLIR-backed implementation.
  void compile(bool Verify) override { M.compile(Verify); }
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

std::shared_ptr<ModuleBase> createMLIRModule(const std::string &Target,
                                             const std::string &Code) {
  return std::make_shared<MLIRModule>(Target, Code);
}

} // namespace proteus_python
