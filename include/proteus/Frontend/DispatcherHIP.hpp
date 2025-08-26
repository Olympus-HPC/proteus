#ifndef PROTEUS_FRONTEND_DISPATCHER_HIP_HPP
#define PROTEUS_FRONTEND_DISPATCHER_HIP_HPP

#if PROTEUS_ENABLE_HIP

#include "proteus/Error.h"
#include "proteus/Frontend/Dispatcher.hpp"
#include "proteus/JitEngineDeviceHIP.hpp"

namespace proteus {

class DispatcherHIP : public Dispatcher {
public:
  static DispatcherHIP &instance() {
    static DispatcherHIP D;
    return D;
  }

  std::unique_ptr<MemoryBuffer>
  compile([[maybe_unused]] std::unique_ptr<LLVMContext> Ctx,
          std::unique_ptr<Module> M, HashT ModuleHash) override {
    std::unique_ptr<MemoryBuffer> ObjectModule = Jit.compileOnly(*M);
    if (!ObjectModule)
      PROTEUS_FATAL_ERROR("Expected non-null object library");

    StorageCache.store(ModuleHash, ObjectModule->getMemBufferRef());

    return ObjectModule;
  }

  std::unique_ptr<MemoryBuffer> lookupObjectModule(HashT ModuleHash) override {
    return StorageCache.lookup(ModuleHash);
  }

  DispatchResult launch(void *KernelFunc, LaunchDims GridDim,
                        LaunchDims BlockDim, ArrayRef<void *> KernelArgs,
                        uint64_t ShmemSize, void *Stream) override {
    dim3 HipGridDim = {GridDim.X, GridDim.Y, GridDim.Z};
    dim3 HipBlockDim = {BlockDim.X, BlockDim.Y, BlockDim.Z};
    hipStream_t HipStream = reinterpret_cast<hipStream_t>(Stream);

    void **KernelArgsPtrs = const_cast<void **>(KernelArgs.data());
    return proteus::launchKernelFunction(
        reinterpret_cast<hipFunction_t>(KernelFunc), HipGridDim, HipBlockDim,
        KernelArgsPtrs, ShmemSize, HipStream);
  }

  StringRef getTargetArch() const override { return Jit.getDeviceArch(); }

  ~DispatcherHIP() {
    CodeCache.printStats();
    StorageCache.printStats();
  }

  void *
  getFunctionAddress(StringRef KernelName,
                     std::optional<MemoryBufferRef> ObjectModule) override {
    auto GetKernelFunc = [&]() {
      // Hash the kernel name to get a unique id.
      HashT HashValue = hash(KernelName);

      if (auto KernelFunc = CodeCache.lookup(HashValue))
        return KernelFunc;

      auto KernelFunc = proteus::getKernelFunctionFromImage(
          KernelName, ObjectModule->getBufferStart(),
          /*RelinkGlobalsByCopy*/ false,
          /* VarNameToDevPtr */ {});

      CodeCache.insert(HashValue, KernelFunc, KernelName);

      return KernelFunc;
    };

    auto KernelFunc = GetKernelFunc();
    return KernelFunc;
  }

private:
  JitEngineDeviceHIP &Jit;
  DispatcherHIP() : Jit(JitEngineDeviceHIP::instance()) {
    TargetModel = TargetModelType::HIP;
  }
  JitCache<hipFunction_t> CodeCache;
  JitStorageCache<hipFunction_t> StorageCache;
};

} // namespace proteus

#endif

#endif // PROTEUS_FRONTEND_DISPATCHER_HIP_HPP
