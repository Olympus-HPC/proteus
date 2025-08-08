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

  void compile([[maybe_unused]] std::unique_ptr<LLVMContext> Ctx,
               std::unique_ptr<Module> M) override {
    Timer T;
    SmallString<4096> Bitcode;
    raw_svector_ostream OS(Bitcode);
    WriteBitcodeToFile(*M, OS);

    HashT HashValue = hash(StringRef{Bitcode.data(), Bitcode.size()});
    auto StoredObject = StorageCache.lookup(HashValue);
    if (StoredObject) {
      Library = std::move(StoredObject);
      return;
    }

    Library = Jit.compileOnly(*M);
    if (!Library)
      PROTEUS_FATAL_ERROR("Expected non-null object library");

    StorageCache.store(HashValue, Library->getMemBufferRef());
  }

  DispatchResult launch(StringRef KernelName, LaunchDims GridDim,
                        LaunchDims BlockDim, ArrayRef<void *> KernelArgs,
                        uint64_t ShmemSize, void *Stream) override {
    auto GetKernelFunc = [&]() {
      HashT HashValue = hash(KernelName);

      if (auto KernelFunc = CodeCache.lookup(HashValue))
        return KernelFunc;

      auto KernelFunc = proteus::getKernelFunctionFromImage(
          KernelName, Library->getBufferStart(),
          /*RelinkGlobalsByCopy*/ false,
          /* VarNameToDevPtr */ {});

      CodeCache.insert(HashValue, KernelFunc, KernelName);

      return KernelFunc;
    };

    auto KernelFunc = GetKernelFunc();

    dim3 HipGridDim = {GridDim.X, GridDim.Y, GridDim.Z};
    dim3 HipBlockDim = {BlockDim.X, BlockDim.Y, BlockDim.Z};
    hipStream_t HipStream = reinterpret_cast<hipStream_t>(Stream);

    void **KernelArgsPtrs = const_cast<void **>(KernelArgs.data());
    return proteus::launchKernelFunction(KernelFunc, HipGridDim, HipBlockDim,
                                         KernelArgsPtrs, ShmemSize, HipStream);
  }

  ~DispatcherHIP() {
    CodeCache.printStats();
    StorageCache.printStats();
  }

protected:
  void *getFunctionAddress(StringRef) override {
    PROTEUS_FATAL_ERROR("HIP does not support getFunctionAddress");
  }

private:
  JitEngineDeviceHIP &Jit;
  DispatcherHIP() : Jit(JitEngineDeviceHIP::instance()) {}
  JitCache<hipFunction_t> CodeCache;
  JitStorageCache<hipFunction_t> StorageCache;
};

} // namespace proteus

#endif

#endif // PROTEUS_FRONTEND_DISPATCHER_HIP_HPP
