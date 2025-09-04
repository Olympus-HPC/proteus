#ifndef PROTEUS_FRONTEND_DISPATCHER_HPP
#define PROTEUS_FRONTEND_DISPATCHER_HPP

#include <llvm/IR/Module.h>
#include <llvm/Support/MemoryBuffer.h>
#include <memory>

#include "proteus/CompiledLibrary.hpp"
#include "proteus/Frontend/TargetModel.hpp"
#include "proteus/Hashing.hpp"

#if PROTEUS_ENABLE_HIP && __HIP__
#include <hip/hip_runtime.h>
#endif

struct LaunchDims {
  unsigned X = 1, Y = 1, Z = 1;
};

namespace proteus {

template <typename T> struct sig_traits;

template <typename R, typename... Args> struct sig_traits<R(Args...)> {
  using return_type = R;
  using argument_types = std::tuple<Args...>;
};

using namespace llvm;

// in Dispatcher.hpp (or a new Errors.hpp)
struct DispatchResult {
  int Ret;

  // construct from an integer error‐code
  constexpr DispatchResult(int Ret = 0) noexcept : Ret(Ret) {}

  // implicit conversion back to int
  operator int() const noexcept { return Ret; }

#if PROTEUS_ENABLE_HIP && __HIP__
  operator hipError_t() const noexcept { return static_cast<hipError_t>(Ret); }
#endif

#if PROTEUS_ENABLE_CUDA && defined(__CUDACC__)
  operator cudaError_t() const noexcept {
    return static_cast<cudaError_t>(Ret);
  }
#endif
};

struct DispatchResult;

class Dispatcher {
protected:
  TargetModelType TargetModel;

public:
  static Dispatcher &getDispatcher(TargetModelType TargetModel);

  virtual std::unique_ptr<MemoryBuffer>
  compile(std::unique_ptr<LLVMContext> Ctx, std::unique_ptr<Module> M,
          HashT ModuleHash) = 0;

  virtual std::unique_ptr<CompiledLibrary>
  lookupCompiledLibrary(HashT ModuleHash) = 0;

  virtual DispatchResult launch(void *KernelFunc, LaunchDims GridDim,
                                LaunchDims BlockDim,
                                ArrayRef<void *> KernelArgs, uint64_t ShmemSize,
                                void *Stream) = 0;

  virtual StringRef getDeviceArch() const = 0;

  template <typename Sig, typename... ArgT>
  typename sig_traits<Sig>::return_type run(void *FuncPtr, ArgT &&...Args) {
    if (!isHostTargetModel(TargetModel))
      PROTEUS_FATAL_ERROR(
          "Dispatcher run interface is only supported for host derived models");

    auto Fn = reinterpret_cast<Sig *>(FuncPtr);
    using Ret = typename sig_traits<Sig>::return_type;

    if constexpr (std::is_void_v<Ret>) {
      Fn(std::forward<ArgT>(Args)...);
      return;
    } else
      return Fn(std::forward<ArgT>(Args)...);
  }

  virtual void *getFunctionAddress(StringRef FunctionName, HashT ModuleHash,
                                   CompiledLibrary &Library) = 0;

  virtual void registerDynamicLibrary(HashT HashValue,
                                      const SmallString<128> &Path) = 0;
};

} // namespace proteus

#endif
