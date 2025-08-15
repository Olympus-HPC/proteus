#ifndef PROTEUS_FRONTEND_DISPATCHER_HPP
#define PROTEUS_FRONTEND_DISPATCHER_HPP

#include <llvm/IR/Module.h>
#include <llvm/Support/MemoryBuffer.h>
#include <memory>

#include "proteus/Frontend/TargetModel.hpp"
#include "proteus/Hashing.hpp"

#if PROTEUS_ENABLE_HIP && __HIP__
#include <hip/hip_runtime.h>
#endif

struct LaunchDims {
  unsigned X = 1, Y = 1, Z = 1;
};

namespace proteus {

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

  virtual std::unique_ptr<MemoryBuffer>
  lookupObjectModule(HashT ModuleHash) = 0;

  virtual DispatchResult
  launch(StringRef KernelName, LaunchDims GridDim, LaunchDims BlockDim,
         ArrayRef<void *> KernelArgs, uint64_t ShmemSize, void *Stream,
         std::optional<MemoryBufferRef> ObjectModule) = 0;

  virtual DispatchResult launch(void *KernelFunc, LaunchDims GridDim,
                                LaunchDims BlockDim,
                                ArrayRef<void *> KernelArgs, uint64_t ShmemSize,
                                void *Stream) = 0;

  virtual StringRef getTargetArch() const = 0;

  // Accepts both a return type or a function signature (needed for C++
  // reference arguments) and disambiguates at compile time.
  template <typename RetOrSig, typename... ArgT>
  auto run(StringRef FuncName, std::optional<MemoryBufferRef> ObjectModule,
           ArgT &&...Args) {
    if (TargetModel != TargetModelType::HOST)
      PROTEUS_FATAL_ERROR("Dispatcher run interface is only support for host");

    void *Addr = getFunctionAddress(FuncName, ObjectModule);

    if constexpr (std::is_function_v<RetOrSig>) {
      auto Fn = reinterpret_cast<RetOrSig *>(Addr);
      using Ret = std::invoke_result_t<RetOrSig, ArgT...>;

      if constexpr (std::is_void_v<Ret>) {
        Fn(std::forward<ArgT>(Args)...);
        return;
      } else
        return Fn(std::forward<ArgT>(Args)...);
    } else {
      using FnPtr = RetOrSig (*)(std::decay_t<ArgT>...);
      auto Fn = reinterpret_cast<FnPtr>(Addr);

      if constexpr (std::is_void_v<RetOrSig>) {
        Fn(std::forward<ArgT>(Args)...);
        return;
      } else
        return Fn(std::forward<ArgT>(Args)...);
    }
  }

  virtual void *
  getFunctionAddress(StringRef FunctionName,
                     std::optional<MemoryBufferRef> ObjectModule) = 0;
};

} // namespace proteus

#endif
