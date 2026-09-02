#ifndef PROTEUS_FRONTEND_DISPATCHER_H
#define PROTEUS_FRONTEND_DISPATCHER_H

#include "proteus/Error.h"
#include "proteus/Frontend/KernelName.h"
#include "proteus/Frontend/TargetModel.h"

#if PROTEUS_ENABLE_HIP && __HIP__
#include <hip/hip_runtime.h>
#endif

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>

namespace llvm {
class LLVMContext;
class Module;
class MemoryBuffer;
class MemoryBufferRef;
class StringRef;
template <typename T> class SmallPtrSetImpl;
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
class CodeGenerationConfig;
struct GlobalVarInfo;

template <typename T> struct sig_traits;

template <typename R, typename... Args> struct sig_traits<R(Args...)> {
  using return_type = R;
  using argument_types = std::tuple<Args...>;
};

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

// CompileOptions controls how a Dispatcher turns a module into an object. The
// defaults match the frontend JIT modules and the annotation runtime overrides
// them.
struct CompileOptions {
  bool DisableIROpt = false;
  // Host dispatchers ignore this option.
  bool LinkDeviceLibraries = true;
  // A null configuration selects Config::get().getCGConfig().
  const CodeGenerationConfig *CGConfig = nullptr;
  // These are prelinked fat binaries, which CUDA produces for RDC.
  llvm::SmallPtrSetImpl<void *> *GlobalLinkedBinaries = nullptr;
  // These globals relink the object against the ones the host program uses.
  const std::unordered_map<std::string, GlobalVarInfo> *VarNameToGlobalInfo =
      nullptr;
  // Setting this relinks globals when loading the image rather than by
  // patching the object.
  bool RelinkGlobalsByCopy = false;
  // Proteus invokes this after IR optimization and before codegen.
  std::function<void(llvm::Module &)> OnOptimized;
};

class Dispatcher {
protected:
  TargetModelType TargetModel;
  const std::string Label;
  std::unique_ptr<ObjectCacheChain> ObjectCache;

  Dispatcher(const std::string &Name, TargetModelType TM);

  void printObjectCacheStats();

public:
  static Dispatcher &getDispatcher(TargetModelType TargetModel);
  virtual ~Dispatcher();

  const std::string &getLabel() const { return Label; }

  // compileModule touches no cache, so it is safe on a worker thread.
  virtual std::unique_ptr<llvm::MemoryBuffer>
  compileModule(llvm::Module &M, const CompileOptions &Opts) = 0;

  std::unique_ptr<llvm::MemoryBuffer>
  compile(std::unique_ptr<llvm::LLVMContext> Ctx,
          std::unique_ptr<llvm::Module> M, const HashT &ModuleHash,
          const CompileOptions &Opts = CompileOptions{});

  std::unique_ptr<CompiledLibrary>
  lookupCompiledLibrary(const HashT &ModuleHash);

  virtual DispatchResult launch(void *KernelFunc, LaunchDims GridDim,
                                LaunchDims BlockDim, void *KernelArgs[],
                                uint64_t ShmemSize, void *Stream) = 0;

  virtual llvm::StringRef getDeviceArch() const = 0;

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

  virtual void *lookupFunction(const KernelName &Name,
                               const HashT &ModuleHash) = 0;

  virtual void *insertFunction(const KernelName &Name, const HashT &ModuleHash,
                               CompiledLibrary &Library) = 0;

  void *getOrInsertFunction(const KernelName &Name, const HashT &ModuleHash,
                            CompiledLibrary &Library);

  virtual void registerDynamicLibrary(const HashT &HashValue,
                                      const std::string &Path) = 0;

  void registerObject(const HashT &HashValue, const llvm::MemoryBufferRef &Obj);
};

} // namespace proteus

#endif
