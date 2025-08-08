#ifndef PROTEUS_FRONTEND_DISPATCHER_CUDA_HPP
#define PROTEUS_FRONTEND_DISPATCHER_CUDA_HPP

#if PROTEUS_ENABLE_CUDA

#include "proteus/Frontend/Dispatcher.hpp"
#include "proteus/JitEngineDeviceCUDA.hpp"

namespace proteus {

class DispatcherCUDA : public Dispatcher {
public:
  static DispatcherCUDA &instance() {
    static DispatcherCUDA D;
    return D;
  }

  void compile([[maybe_unused]] std::unique_ptr<LLVMContext> Ctx,
               std::unique_ptr<Module> M) override {
    SmallString<4096> Bitcode;
    raw_svector_ostream OS(Bitcode);
    WriteBitcodeToFile(*M, OS);

    HashT HashValue = hash(StringRef{Bitcode.data(), Bitcode.size()});
    auto StoredObject = StorageCache.lookup(HashValue);
    if (StoredObject) {
      Library = std::move(StoredObject);
      return;
    }

    // CMake finds LIBDEVICE_BC_PATH.
    auto LibDeviceBuffer = llvm::MemoryBuffer::getFile(LIBDEVICE_BC_PATH);
    auto LibDeviceModule = llvm::parseBitcodeFile(
        LibDeviceBuffer->get()->getMemBufferRef(), M->getContext());

    llvm::Linker linker(*M);
    linker.linkInModule(std::move(LibDeviceModule.get()));

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

    dim3 CudaGridDim = {GridDim.X, GridDim.Y, GridDim.Z};
    dim3 CudaBlockDim = {BlockDim.X, BlockDim.Y, BlockDim.Z};
    cudaStream_t CudaStream = reinterpret_cast<cudaStream_t>(Stream);

    void **KernelArgsPtrs = const_cast<void **>(KernelArgs.data());
    return proteus::launchKernelFunction(KernelFunc, CudaGridDim, CudaBlockDim,
                                         KernelArgsPtrs, ShmemSize, CudaStream);
  }

  ~DispatcherCUDA() {
    CodeCache.printStats();
    StorageCache.printStats();
  }

protected:
  void *getFunctionAddress(StringRef) override {
    PROTEUS_FATAL_ERROR("CUDA does not support getFunctionAddress");
  }

private:
  JitEngineDeviceCUDA &Jit;
  DispatcherCUDA() : Jit(JitEngineDeviceCUDA::instance()) {}
  JitCache<CUfunction> CodeCache;
  JitStorageCache<CUfunction> StorageCache;
};

} // namespace proteus

#endif

#endif // PROTEUS_FRONTEND_DISPATCHER_CUDA_HPP
