#ifndef PROTEUS_FRONTEND_DISPATCHER_HPP
#define PROTEUS_FRONTEND_DISPATCHER_HPP

#include <llvm/IR/Module.h>
#include <llvm/Support/MemoryBuffer.h>
#include <memory>

#if PROTEUS_ENABLE_HIP && __HIP__
#include <hip/hip_runtime.h>
#endif

enum class TargetModelType { HOST, CUDA, HIP };

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
};

struct DispatchResult;

class Dispatcher {
public:
  static Dispatcher &getDispatcher(TargetModelType Model);

  virtual void compile(std::unique_ptr<Module> M) = 0;

  virtual DispatchResult launch(StringRef KernelName, LaunchDims GridDim,
                                LaunchDims BlockDim,
                                ArrayRef<void *> KernelArgs, uint64_t ShmemSize,
                                void *Stream) = 0;

  template <typename Ret, typename... ArgT>
  Ret run(StringRef FuncName, ArgT &&...Args) {
    void *Addr = getFunctionAddress(FuncName);
    using FnPtr = Ret (*)(ArgT...);
    auto Fn = reinterpret_cast<FnPtr>(Addr);
    return Fn(std::forward<ArgT>(Args)...);
  }

protected:
  std::unique_ptr<MemoryBuffer> Library = nullptr;
  virtual void *getFunctionAddress(StringRef FunctionName) = 0;
};

} // namespace proteus

#endif