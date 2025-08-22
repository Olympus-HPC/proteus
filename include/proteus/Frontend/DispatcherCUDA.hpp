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

  std::unique_ptr<MemoryBuffer>
  compile([[maybe_unused]] std::unique_ptr<LLVMContext> Ctx,
          std::unique_ptr<Module> M, HashT ModuleHash) override {

    // CMake finds LIBDEVICE_BC_PATH.
    auto LibDeviceBuffer = llvm::MemoryBuffer::getFile(LIBDEVICE_BC_PATH);
    auto LibDeviceModule = llvm::parseBitcodeFile(
        LibDeviceBuffer->get()->getMemBufferRef(), M->getContext());

    llvm::Linker linker(*M);
    linker.linkInModule(std::move(LibDeviceModule.get()));

    std::unique_ptr<MemoryBuffer> ObjectModule = Jit.compileOnly(*M);
    if (!ObjectModule)
      PROTEUS_FATAL_ERROR("Expected non-null object library");

    StorageCache.store(ModuleHash, ObjectModule->getMemBufferRef());

    return ObjectModule;
  }

  std::unique_ptr<MemoryBuffer> lookupObjectModule(HashT ModuleHash) override {
    return StorageCache.lookup(ModuleHash);
  }

  DispatchResult launch(StringRef KernelName, LaunchDims GridDim,
                        LaunchDims BlockDim, ArrayRef<void *> KernelArgs,
                        uint64_t ShmemSize, void *Stream,
                        std::optional<MemoryBufferRef> ObjectModule) override {
    auto *KernelFunc = getFunctionAddress(KernelName, ObjectModule);

    dim3 CudaGridDim = {GridDim.X, GridDim.Y, GridDim.Z};
    dim3 CudaBlockDim = {BlockDim.X, BlockDim.Y, BlockDim.Z};
    cudaStream_t CudaStream = reinterpret_cast<cudaStream_t>(Stream);

    void **KernelArgsPtrs = const_cast<void **>(KernelArgs.data());
    return proteus::launchKernelFunction(
        reinterpret_cast<cudaFunction_t>(KernelFunc), CudaGridDim, CudaBlockDim,
        KernelArgsPtrs, ShmemSize, CudaStream);
  }

  DispatchResult launch(void *KernelFunc, LaunchDims GridDim,
                        LaunchDims BlockDim, ArrayRef<void *> KernelArgs,
                        uint64_t ShmemSize, void *Stream) override {
    dim3 CudaGridDim = {GridDim.X, GridDim.Y, GridDim.Z};
    dim3 CudaBlockDim = {BlockDim.X, BlockDim.Y, BlockDim.Z};
    cudaStream_t CudaStream = reinterpret_cast<cudaStream_t>(Stream);

    void **KernelArgsPtrs = const_cast<void **>(KernelArgs.data());
    return proteus::launchKernelFunction(
        reinterpret_cast<cudaFunction_t>(KernelFunc), CudaGridDim, CudaBlockDim,
        KernelArgsPtrs, ShmemSize, CudaStream);
  }

  StringRef getTargetArch() const override { return Jit.getDeviceArch(); }

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

  ~DispatcherCUDA() {
    CodeCache.printStats();
    StorageCache.printStats();
  }

private:
  JitEngineDeviceCUDA &Jit;
  DispatcherCUDA() : Jit(JitEngineDeviceCUDA::instance()) {
    TargetModel = TargetModelType::CUDA;
  }
  JitCache<CUfunction> CodeCache;
  JitStorageCache<CUfunction> StorageCache;
};

} // namespace proteus

#endif

#endif // PROTEUS_FRONTEND_DISPATCHER_CUDA_HPP
