#ifndef PROTEUS_MLIRJITMODULE_H
#define PROTEUS_MLIRJITMODULE_H

#include "proteus/Frontend/Dispatcher.h"
#include "proteus/Frontend/TargetModel.h"
#include "proteus/JitFuncAttribute.h"

#include <memory>
#include <string>
#include <type_traits>
#include <utility>

namespace proteus {

struct CompiledLibrary;
class HashT;

class MLIRJitModule {
private:
  TargetModelType TargetModel;
  std::string Code;
  Dispatcher &Dispatch;
  std::unique_ptr<HashT> ModuleHash;
  std::unique_ptr<CompiledLibrary> Library;
  bool IsCompiled = false;

public:
  explicit MLIRJitModule(TargetModelType TargetModel, const std::string &Code);
  explicit MLIRJitModule(const std::string &Target, const std::string &Code);
  ~MLIRJitModule();

  void compile(bool Verify = false);

  // Expose the target model so higher-level bindings can reject invalid API
  // combinations before dispatch.
  TargetModelType getTargetModel() const { return TargetModel; }

  void setFuncAttribute(void *KernelFunc, JitFuncAttribute Attr, int Value);
  void *getFunctionAddress(const std::string &Name);
  void *getKernelAddress(const std::string &Name) {
    if (!IsCompiled)
      compile();

    if (TargetModel == TargetModelType::HOST)
      reportFatalError(
          "Error: getKernelAddress() applies only to device modules");

    // Kernel symbols are loaded through the same compiled image as host
    // function symbols once target validation is complete.
    return getFunctionAddress(Name);
  }

  DispatchResult launch(void *KernelFunc, LaunchDims GridDim,
                        LaunchDims BlockDim, void *KernelArgs[],
                        uint64_t ShmemSize, void *Stream);

  CompiledLibrary &getLibrary() {
    if (!IsCompiled)
      compile();

    if (!Library)
      reportFatalError("Expected non-null library after compilation");

    return *Library;
  }

  template <typename Sig> struct FunctionHandle;
  template <typename RetT, typename... ArgT>
  struct FunctionHandle<RetT(ArgT...)> {
    MLIRJitModule &M;
    void *FuncPtr;

    explicit FunctionHandle(MLIRJitModule &M, void *FuncPtr)
        : M(M), FuncPtr(FuncPtr) {}

    RetT run(ArgT... Args) {
      if constexpr (std::is_void_v<RetT>) {
        M.Dispatch.template run<RetT(ArgT...)>(FuncPtr,
                                               std::forward<ArgT>(Args)...);
      } else {
        return M.Dispatch.template run<RetT(ArgT...)>(
            FuncPtr, std::forward<ArgT>(Args)...);
      }
    }
  };

  template <typename Sig> struct KernelHandle;
  template <typename RetT, typename... ArgT>
  struct KernelHandle<RetT(ArgT...)> {
    MLIRJitModule &M;
    void *FuncPtr = nullptr;

    explicit KernelHandle(MLIRJitModule &M, void *FuncPtr)
        : M(M), FuncPtr(FuncPtr) {
      static_assert(std::is_void_v<RetT>, "Kernel function must return void");
    }

    void setFuncAttribute(JitFuncAttribute Attr, int Value) {
      M.setFuncAttribute(FuncPtr, Attr, Value);
    }

    auto launch(LaunchDims GridDim, LaunchDims BlockDim, uint64_t ShmemSize,
                void *Stream, ArgT... Args) {
      void *Ptrs[sizeof...(ArgT)] = {(void *)&Args...};
      return M.launch(FuncPtr, GridDim, BlockDim, Ptrs, ShmemSize, Stream);
    }
  };

  template <typename Sig>
  FunctionHandle<Sig> getFunction(const std::string &Name) {
    if (!IsCompiled)
      compile();

    if (!isHostTargetModel(TargetModel))
      reportFatalError("Error: getFunction() applies only to host modules");

    void *FuncPtr = getFunctionAddress(Name);
    return FunctionHandle<Sig>(*this, FuncPtr);
  }

  template <typename Sig> KernelHandle<Sig> getKernel(const std::string &Name) {
    if (!IsCompiled)
      compile();

    if (TargetModel == TargetModelType::HOST)
      reportFatalError("Error: getKernel() applies only to device modules");

    void *FuncPtr = getFunctionAddress(Name);
    return KernelHandle<Sig>(*this, FuncPtr);
  }
};

} // namespace proteus

#endif // PROTEUS_MLIRJITMODULE_H
