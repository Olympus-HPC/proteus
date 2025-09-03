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
          std::unique_ptr<Module> Mod, HashT ModuleHash) override {
    // This is necessary to ensure Ctx outlives M. Setting [[maybe_unused]] can
    // trigger a lifetime bug.
    auto CtxOwner = std::move(Ctx);
    auto ModOwner = std::move(Mod);

    // CMake finds LIBDEVICE_BC_PATH.
    auto LibDeviceBuffer = llvm::MemoryBuffer::getFile(LIBDEVICE_BC_PATH);
    auto LibDeviceModule = llvm::parseBitcodeFile(
        LibDeviceBuffer->get()->getMemBufferRef(), ModOwner->getContext());

    llvm::Linker linker(*ModOwner);
    linker.linkInModule(std::move(LibDeviceModule.get()));

    std::unique_ptr<MemoryBuffer> ObjectModule = Jit.compileOnly(*ModOwner);
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
    dim3 CudaGridDim = {GridDim.X, GridDim.Y, GridDim.Z};
    dim3 CudaBlockDim = {BlockDim.X, BlockDim.Y, BlockDim.Z};
    cudaStream_t CudaStream = reinterpret_cast<cudaStream_t>(Stream);

    void **KernelArgsPtrs = const_cast<void **>(KernelArgs.data());
    return proteus::launchKernelFunction(
        reinterpret_cast<cudaFunction_t>(KernelFunc), CudaGridDim, CudaBlockDim,
        KernelArgsPtrs, ShmemSize, CudaStream);
  }

  StringRef getDeviceArch() const override { return Jit.getDeviceArch(); }

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

  void loadDynamicLibrary(const SmallString<128> &) override {
    PROTEUS_FATAL_ERROR(
        "Device dispatcher does not implement loadDynamicLibrary");
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
