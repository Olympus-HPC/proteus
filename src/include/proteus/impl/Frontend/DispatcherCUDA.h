#ifndef PROTEUS_FRONTEND_DISPATCHER_CUDA_H
#define PROTEUS_FRONTEND_DISPATCHER_CUDA_H

#if PROTEUS_ENABLE_CUDA

#include "proteus/Frontend/Dispatcher.h"
#include "proteus/impl/JitEngineDeviceCUDA.h"

namespace proteus {

class DispatcherCUDA : public Dispatcher {
public:
  static DispatcherCUDA &instance() {
    static DispatcherCUDA D;
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

    // CMake finds LIBDEVICE_BC_PATH.
    auto LibDeviceBuffer = llvm::MemoryBuffer::getFile(LIBDEVICE_BC_PATH);
    auto LibDeviceModule = llvm::parseBitcodeFile(
        LibDeviceBuffer->get()->getMemBufferRef(), ModOwner->getContext());

    llvm::Linker linker(*ModOwner);
    linker.linkInModule(std::move(LibDeviceModule.get()),
                        llvm::Linker::Flags::LinkOnlyNeeded);

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
    dim3 CudaGridDim = {GridDim.X, GridDim.Y, GridDim.Z};
    dim3 CudaBlockDim = {BlockDim.X, BlockDim.Y, BlockDim.Z};
    cudaStream_t CudaStream = reinterpret_cast<cudaStream_t>(Stream);

    return proteus::launchKernelFunction(
        reinterpret_cast<cudaFunction_t>(KernelFunc), CudaGridDim, CudaBlockDim,
        KernelArgs, ShmemSize, CudaStream);
  }

  StringRef getDeviceArch() const override { return Jit.getDeviceArch(); }

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
    reportFatalError("Dispatch CUDA does not support registerDynamicLibrary");
  }

  ~DispatcherCUDA() {
    CodeCache.printStats();
    getObjectCache().printStats();
  }

private:
  JitEngineDeviceCUDA &Jit;
  DispatcherCUDA()
      : Dispatcher("DispatcherCUDA", TargetModelType::CUDA),
        Jit(JitEngineDeviceCUDA::instance()) {}
  MemoryCache<CUfunction> CodeCache{"DispatcherCUDA"};
};

} // namespace proteus

#endif

#endif // PROTEUS_FRONTEND_DISPATCHER_CUDA_H
