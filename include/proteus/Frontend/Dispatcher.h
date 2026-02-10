#ifndef PROTEUS_FRONTEND_DISPATCHER_H
#define PROTEUS_FRONTEND_DISPATCHER_H

#include "proteus/Error.h"
#include "proteus/Frontend/TargetModel.h"

#if PROTEUS_ENABLE_HIP && __HIP__
#include <hip/hip_runtime.h>
#endif

#include <cstdint>
#include <memory>
#include <type_traits>

namespace llvm {
class LLVMContext;
class Module;
class MemoryBuffer;
} // namespace llvm

struct LaunchDims {
  unsigned X = 1, Y = 1, Z = 1;

  constexpr LaunchDims() = default;

  constexpr LaunchDims(unsigned X, unsigned Y = 1, unsigned Z = 1)
      : X(X), Y(Y), Z(Z) {}

  // Templated converting constructor for dim3-like types.
  template <
      typename T,
      typename = std::enable_if_t<
          std::is_convertible_v<decltype(std::declval<T>().x), unsigned> &&
          std::is_convertible_v<decltype(std::declval<T>().y), unsigned> &&
          std::is_convertible_v<decltype(std::declval<T>().z), unsigned>>>
  constexpr LaunchDims(const T &Dims) : X(Dims.x), Y(Dims.y), Z(Dims.z) {}
};

namespace proteus {

class ObjectCacheChain;
struct CompiledLibrary;
class HashT;

template <typename T> struct sig_traits;

template <typename R, typename... Args> struct sig_traits<R(Args...)> {
  using return_type = R;
  using argument_types = std::tuple<Args...>;
};

using namespace llvm;

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
  ObjectCacheChain *Cache = nullptr;

  Dispatcher(const std::string &Name, TargetModelType TM);

  ObjectCacheChain &getObjectCache();

public:
  static Dispatcher &getDispatcher(TargetModelType TargetModel);
  virtual ~Dispatcher() = default;

  virtual std::unique_ptr<MemoryBuffer>
  compile(std::unique_ptr<LLVMContext> Ctx, std::unique_ptr<Module> M,
          const HashT &ModuleHash, bool DisableIROpt = false) = 0;

  virtual std::unique_ptr<CompiledLibrary>
  lookupCompiledLibrary(const HashT &ModuleHash) = 0;

  virtual DispatchResult launch(void *KernelFunc, LaunchDims GridDim,
                                LaunchDims BlockDim, void *KernelArgs[],
                                uint64_t ShmemSize, void *Stream) = 0;

  virtual StringRef getDeviceArch() const = 0;

  template <typename Sig, typename... ArgT>
  typename sig_traits<Sig>::return_type run(void *FuncPtr, ArgT &&...Args) {
    if (!isHostTargetModel(TargetModel))
      reportFatalError(
          "Dispatcher run interface is only supported for host derived models");

    auto Fn = reinterpret_cast<Sig *>(FuncPtr);
    using Ret = typename sig_traits<Sig>::return_type;

    if constexpr (std::is_void_v<Ret>) {
      Fn(std::forward<ArgT>(Args)...);
      return;
    } else
      return Fn(std::forward<ArgT>(Args)...);
  }

  virtual void *getFunctionAddress(const std::string &FunctionName,
                                   const HashT &ModuleHash,
                                   CompiledLibrary &Library) = 0;

  virtual void registerDynamicLibrary(const HashT &HashValue,
                                      const std::string &Path) = 0;
};

} // namespace proteus

#endif
