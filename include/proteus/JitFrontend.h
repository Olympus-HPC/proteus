#ifndef PROTEUS_JIT_DEV_H
#define PROTEUS_JIT_DEV_H

#include "proteus/Error.h"
#include "proteus/Frontend/Dispatcher.h"
#include "proteus/Frontend/Func.h"
#include "proteus/Frontend/LoopNest.h"
#include "proteus/Frontend/TypeMap.h"
#include "proteus/Init.h"

#include <deque>
#include <type_traits>

namespace proteus {

struct CompiledLibrary;

class JitModule {
private:
  std::unique_ptr<LLVMCodeBuilder> CB;
  std::unique_ptr<CompiledLibrary> Library;

  std::deque<std::unique_ptr<FuncBase>> Functions;
  TargetModelType TargetModel;
  std::string TargetTriple;
  Dispatcher &Dispatch;

  std::unique_ptr<HashT> ModuleHash;
  bool IsCompiled = false;

  template <typename... ArgT> struct KernelHandle;

  template <typename RetT, typename... ArgT>
  Func<RetT, ArgT...> &buildFuncFromArgsList(const std::string &Name,
                                             ArgTypeList<ArgT...>) {
    auto TypedFn =
        std::make_unique<Func<RetT, ArgT...>>(*this, *CB, Name, Dispatch);
    Func<RetT, ArgT...> &TypedFnRef = *TypedFn;
    Functions.emplace_back(std::move(TypedFn));
    TypedFnRef.declArgs();
    return TypedFnRef;
  }

  template <typename... ArgT>
  KernelHandle<ArgT...> buildKernelFromArgsList(const std::string &Name,
                                                ArgTypeList<ArgT...>) {
    auto TypedFn =
        std::make_unique<Func<void, ArgT...>>(*this, *CB, Name, Dispatch);
    Func<void, ArgT...> &TypedFnRef = *TypedFn;
    TypedFn->declArgs();

#if PROTEUS_ENABLE_CUDA || PROTEUS_ENABLE_HIP
    std::unique_ptr<FuncBase> &Fn = Functions.emplace_back(std::move(TypedFn));
    Fn->setKernel();
#else
    reportFatalError("setKernel() is only supported for CUDA/HIP");
#endif
    return KernelHandle<ArgT...>{TypedFnRef, *this};
  }

  template <typename... ArgT> struct KernelHandle {
    Func<void, ArgT...> &F;
    JitModule &M;

    void setLaunchBounds([[maybe_unused]] int MaxThreadsPerBlock,
                         [[maybe_unused]] int MinBlocksPerSM = 0) {
      if (!M.isDeviceModule())
        reportFatalError("Expected a device module for setLaunchBounds");

      if (M.isCompiled())
        reportFatalError("setLaunchBounds must be called before compile()");

#if PROTEUS_ENABLE_CUDA || PROTEUS_ENABLE_HIP
      F.setLaunchBoundsForKernel(MaxThreadsPerBlock, MinBlocksPerSM);
#else
      reportFatalError("Unsupported target for setLaunchBounds");
#endif
    }

    // Launch with type-safety.
    [[nodiscard]] auto launch(LaunchDims Grid, LaunchDims Block,
                              uint64_t ShmemBytes, void *Stream, ArgT... Args) {
      // Pointers to the local parameter copies.
      void *Ptrs[sizeof...(ArgT)] = {(void *)&Args...};

      if (!M.isCompiled())
        M.compile();

      auto GetKernelFunc = [&]() {
        // Get the kernel func pointer directly from the Func object if
        // available.
        if (auto KernelFunc = F.getCompiledFunc()) {
          return KernelFunc;
        }

        // Get the kernel func pointer from the Dispatch and store it to the
        // Func object to avoid cache lookups.
        // TODO: Re-think caching and dispatchers.
        auto KernelFunc = reinterpret_cast<decltype(F.getCompiledFunc())>(
            M.Dispatch.getFunctionAddress(F.getName(), M.getModuleHash(),
                                          M.getLibrary()));

        F.setCompiledFunc(KernelFunc);

        return KernelFunc;
      };

      return M.Dispatch.launch(reinterpret_cast<void *>(GetKernelFunc()), Grid,
                               Block, Ptrs, ShmemBytes, Stream);
    }

    FuncBase *operator->() { return &F; }
  };

  bool isDeviceModule() {
    return ((TargetModel == TargetModelType::CUDA) ||
            (TargetModel == TargetModelType::HIP));
  }

public:
  JitModule(const std::string &Target = "host");

  // Disable copy and move constructors.
  JitModule(const JitModule &) = delete;
  JitModule &operator=(const JitModule &) = delete;
  JitModule(JitModule &&) = delete;
  JitModule &operator=(JitModule &&) = delete;

  ~JitModule();

  template <typename Sig> auto &addFunction(const std::string &Name) {
    using RetT = typename FnSig<Sig>::RetT;
    using ArgT = typename FnSig<Sig>::ArgsTList;

    if (IsCompiled)
      reportFatalError("The module is compiled, no further code can be added");

    return buildFuncFromArgsList<RetT>(Name, ArgT{});
  }

  bool isCompiled() const { return IsCompiled; }

  const Module &getModule() const { return CB->getModule(); }
  Module &getModule() { return CB->getModule(); }

  template <typename Sig> auto addKernel(const std::string &Name) {
    using RetT = typename FnSig<Sig>::RetT;
    static_assert(std::is_void_v<RetT>, "Kernels must have void return type");
    using ArgT = typename FnSig<Sig>::ArgsTList;

    if (IsCompiled)
      reportFatalError("The module is compiled, no further code can be added");

    if (!isDeviceModule())
      reportFatalError("Expected a device module for addKernel");

    return buildKernelFromArgsList(Name, ArgT{});
  }

  void compile(bool Verify = false);

  const HashT &getModuleHash() const;

  Dispatcher &getDispatcher() const { return Dispatch; }

  TargetModelType getTargetModel() const { return TargetModel; }

  const std::string &getTargetTriple() const { return TargetTriple; }

  CompiledLibrary &getLibrary() {
    if (!IsCompiled)
      compile();

    if (!Library)
      reportFatalError("Expected non-null library after compilation");

    return *Library;
  }

  void print();
};

template <typename RetT, typename... ArgT>
RetT Func<RetT, ArgT...>::operator()(ArgT... Args) {
  if (!J.isCompiled())
    J.compile();

  if (!CompiledFunc) {
    CompiledFunc = reinterpret_cast<decltype(CompiledFunc)>(
        J.getDispatcher().getFunctionAddress(getName(), J.getModuleHash(),
                                             J.getLibrary()));
  }

  if (J.getTargetModel() != TargetModelType::HOST)
    reportFatalError(
        "Target is a GPU model, cannot directly run functions, use launch()");

  if constexpr (std::is_void_v<RetT>)
    Dispatch.run<RetT(ArgT...)>(reinterpret_cast<void *>(CompiledFunc),
                                Args...);
  else
    return Dispatch.run<RetT(ArgT...)>(reinterpret_cast<void *>(CompiledFunc),
                                       Args...);
}

} // namespace proteus

#endif
