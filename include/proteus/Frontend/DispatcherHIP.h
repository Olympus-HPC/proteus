#ifndef PROTEUS_FRONTEND_DISPATCHER_HIP_H
#define PROTEUS_FRONTEND_DISPATCHER_HIP_H

#if PROTEUS_ENABLE_HIP

#include "proteus/Caching/ObjectCacheChain.h"
#include "proteus/Caching/ObjectCacheRegistry.h"
#include "proteus/Error.h"
#include "proteus/Frontend/Dispatcher.h"
#include "proteus/JitEngineDeviceHIP.h"

namespace proteus {

class DispatcherHIP : public Dispatcher {
public:
  static DispatcherHIP &instance() {
    static DispatcherHIP D;
    return D;
  }

  std::unique_ptr<MemoryBuffer> compile(std::unique_ptr<LLVMContext> Ctx,
                                        std::unique_ptr<Module> Mod,
                                        const HashT &ModuleHash,
                                        bool DisableIROpt = false) override {
    // This is necessary to ensure Ctx outlives M. Setting [[maybe_unused]] can
    // trigger a lifetime bug.
    auto CtxOwner = std::move(Ctx);
    auto ModOwner = std::move(Mod);

    std::unique_ptr<MemoryBuffer> ObjectModule =
        Jit.compileOnly(*ModOwner, DisableIROpt);
    if (!ObjectModule)
      reportFatalError("Expected non-null object library");

    getObjectCache().store(
        ModuleHash, CacheEntry::staticObject(ObjectModule->getMemBufferRef()));

    return ObjectModule;
  }

  std::unique_ptr<CompiledLibrary>
  lookupCompiledLibrary(const HashT &ModuleHash) override {
    return getObjectCache().lookup(ModuleHash);
  }

  DispatchResult launch(void *KernelFunc, LaunchDims GridDim,
                        LaunchDims BlockDim, void *KernelArgs[],
                        uint64_t ShmemSize, void *Stream) override {
    dim3 HipGridDim = {GridDim.X, GridDim.Y, GridDim.Z};
    dim3 HipBlockDim = {BlockDim.X, BlockDim.Y, BlockDim.Z};
    hipStream_t HipStream = reinterpret_cast<hipStream_t>(Stream);

    return proteus::launchKernelFunction(
        reinterpret_cast<hipFunction_t>(KernelFunc), HipGridDim, HipBlockDim,
        KernelArgs, ShmemSize, HipStream);
  }

  StringRef getDeviceArch() const override { return Jit.getDeviceArch(); }

  ~DispatcherHIP() {
    CodeCache.printStats();
    getObjectCache().printStats();
  }

  void *getFunctionAddress(const std::string &KernelName,
                           const HashT &ModuleHash,
                           CompiledLibrary &Library) override {
    auto GetKernelFunc = [&]() {
      // Hash the kernel name to get a unique id.
      HashT HashValue = hash(KernelName, ModuleHash);

      if (auto KernelFunc = CodeCache.lookup(HashValue))
        return KernelFunc;

      auto KernelFunc = proteus::getKernelFunctionFromImage(
          KernelName, Library.ObjectModule->getBufferStart(),
          /*RelinkGlobalsByCopy*/ false,
          /* VarNameToGlobalInfo */ {});

      CodeCache.insert(HashValue, KernelFunc, KernelName);

      return KernelFunc;
    };

    auto KernelFunc = GetKernelFunc();
    return KernelFunc;
  }

  void registerDynamicLibrary(const HashT &, const std::string &) override {
    reportFatalError("Dispatch HIP does not support registerDynamicLibrary");
  }

private:
  JitEngineDeviceHIP &Jit;
  DispatcherHIP() : Jit(JitEngineDeviceHIP::instance()) {
    TargetModel = TargetModelType::HIP;
    DispatcherName = "DispatcherHIP";
    ObjectCacheRegistry::instance().create(DispatcherName);
  }
  MemoryCache<hipFunction_t> CodeCache{"DispatcherHIP"};
};

} // namespace proteus

#endif

#endif // PROTEUS_FRONTEND_DISPATCHER_HIP_H
