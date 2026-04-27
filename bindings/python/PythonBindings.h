#ifndef PROTEUS_PYTHON_BINDINGS_H
#define PROTEUS_PYTHON_BINDINGS_H

#include "proteus/Frontend/Dispatcher.h"
#include "proteus/Frontend/TargetModel.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace proteus_python {

// Scalar and pointer element kinds exposed through the Python bindings.
enum class PyType { I8, I32, I64, U32, U64, F32, F64, Ptr };

// Lightweight type descriptor used when marshalling Python-side values.
struct Type {
  PyType Kind;
};

// Common execution interface implemented by the C++ and MLIR-backed modules.
class ModuleBase {
public:
  virtual ~ModuleBase() = default;

  // Lowers and finalizes the module for execution.
  virtual void compile(bool Verify) = 0;

  // Resolves a host-callable symbol emitted by the module.
  virtual void *getFunctionAddress(const std::string &Name) = 0;

  // Reports the compiled module target model for API validation.
  virtual proteus::TargetModelType getTargetModel() const = 0;

  // Resolves a device kernel entry point and rejects host-only modules.
  virtual void *getKernelAddress(const std::string &Name) = 0;

  // Launches a compiled kernel using the provided grid, block, and stream.
  virtual proteus::DispatchResult launch(void *KernelFunc, LaunchDims GridDim,
                                         LaunchDims BlockDim,
                                         void *KernelArgs[], uint64_t ShmemSize,
                                         void *Stream) = 0;
};

// Builds a module from C/C++ source and the requested compiler toolchain.
std::shared_ptr<ModuleBase>
createCppModule(const std::string &Target, const std::string &Code,
                const std::vector<std::string> &ExtraArgs,
                const std::string &Compiler);

#if PROTEUS_ENABLE_MLIR
// Builds a module from MLIR source for the requested backend target.
std::shared_ptr<ModuleBase> createMLIRModule(const std::string &Target,
                                             const std::string &Code);
#endif

} // namespace proteus_python

#endif
