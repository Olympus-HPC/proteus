#ifndef PROTEUS_FRONTEND_DISPATCHER_DEVICE_H
#define PROTEUS_FRONTEND_DISPATCHER_DEVICE_H

#if PROTEUS_ENABLE_HIP || PROTEUS_ENABLE_CUDA

#include "proteus/Error.h"
#include "proteus/Frontend/Dispatcher.h"
#include "proteus/TimeTracing.h"
#include "proteus/impl/Caching/MemoryCache.h"
#include "proteus/impl/Caching/ObjectCacheChain.h"
#include "proteus/impl/CompiledLibrary.h"
#include "proteus/impl/Config.h"
#include "proteus/impl/CoreDevice.h"
#include "proteus/impl/CoreLLVM.h"
#include "proteus/impl/CoreLLVMDevice.h"
#include "proteus/impl/JitEngineDevice.h"

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/Support/MemoryBuffer.h>

namespace proteus {

// DispatcherDevice is the shared implementation of the device dispatchers,
// where JitT is the device JIT engine. Subclasses supply the device library
// linking and the singleton.
template <typename JitT> class DispatcherDevice : public Dispatcher {
public:
  using KernelFunction_t = typename DeviceTraits<JitT>::KernelFunction_t;

  std::unique_ptr<MemoryBuffer>
  compileModule(Module &M, const CompileOptions &Opts) override {
    TIMESCOPE(DispatcherDevice, compileModule);

    const CodeGenerationConfig &CGConfig =
        Opts.CGConfig ? *Opts.CGConfig : Config::get().getCGConfig();

    if (Opts.LinkDeviceLibraries)
      linkDeviceLibraries(M);

    if (Opts.DisableIROpt) {
      if (Config::get().traceSpecializations())
        Logger::trace("[SkipOpt] Skipping JitEngine IR optimization\n");
    } else if (JitT::optimizesBeforeCodegen(CGConfig.codeGenOption())) {
      proteus::optimizeIR(M, Jit.getDeviceArch(),
                          OptimizationPipelineConfig(CGConfig));
    }

    if (Opts.OnOptimized)
      Opts.OnOptimized(M);

    SmallPtrSet<void *, 8> NoLinkedBinaries;
    SmallPtrSetImpl<void *> &GlobalLinkedBinaries =
        Opts.GlobalLinkedBinaries ? *Opts.GlobalLinkedBinaries
                                  : NoLinkedBinaries;

    auto ObjBuf = Jit.codegenObject(M, GlobalLinkedBinaries, CGConfig);
    if (!ObjBuf)
      reportFatalError("Expected non-null object library");

    if (Opts.VarNameToGlobalInfo && !Opts.RelinkGlobalsByCopy)
      proteus::relinkGlobalsObject(ObjBuf->getMemBufferRef(),
                                   *Opts.VarNameToGlobalInfo);

    return ObjBuf;
  }

  DispatchResult launch(void *KernelFunc, LaunchDims GridDim,
                        LaunchDims BlockDim, void *KernelArgs[],
                        uint64_t ShmemSize, void *Stream) override {
    TIMESCOPE(DispatcherDevice, launch);
    dim3 DevGridDim = {GridDim.X, GridDim.Y, GridDim.Z};
    dim3 DevBlockDim = {BlockDim.X, BlockDim.Y, BlockDim.Z};
    auto DevStream =
        reinterpret_cast<typename DeviceTraits<JitT>::DeviceStream_t>(Stream);

    return proteus::launchKernelFunction(
        reinterpret_cast<KernelFunction_t>(KernelFunc), DevGridDim, DevBlockDim,
        KernelArgs, ShmemSize, DevStream);
  }

  StringRef getDeviceArch() const override { return Jit.getDeviceArch(); }

  void *lookupFunction(const std::string &KernelName,
                       const HashT &ModuleHash) override {
    HashT HashValue = hash(KernelName, ModuleHash);
    return CodeCache.lookup(HashValue);
  }

  void *loadFunctionAddress(const std::string &KernelName,
                            const HashT &ModuleHash, CompiledLibrary &Library,
                            const std::string &TraceName = "") override {
    TIMESCOPE(DispatcherDevice, loadFunctionAddress);
    HashT HashValue = hash(KernelName, ModuleHash);

    static const std::unordered_map<std::string, GlobalVarInfo> NoGlobals;
    const auto &VarNameToGlobalInfo =
        Library.VarNameToGlobalInfo ? *Library.VarNameToGlobalInfo : NoGlobals;

    // Objects coming from the object cache have not been relinked against
    // the current process' globals.
    if (Library.VarNameToGlobalInfo && !Library.RelinkGlobalsByCopy &&
        !Library.GlobalsRelinked) {
      proteus::relinkGlobalsObject(Library.ObjectModule->getMemBufferRef(),
                                   VarNameToGlobalInfo);
      Library.GlobalsRelinked = true;
    }

    auto KernelFunc = proteus::getKernelFunctionFromImage(
        KernelName, Library.ObjectModule->getBufferStart(),
        Library.RelinkGlobalsByCopy, VarNameToGlobalInfo);
    Library.IsLoaded = true;

    CodeCache.insert(HashValue, KernelFunc,
                     TraceName.empty() ? KernelName : TraceName);

    return KernelFunc;
  }

  void registerDynamicLibrary(const HashT &, const std::string &) override {
    reportFatalError(Label + " does not support registerDynamicLibrary");
  }

  ~DispatcherDevice() {
    if (Config::get().traceCacheStats())
      CodeCache.printStats();
    CodeCache.printKernelTrace();
    printObjectCacheStats();
  }

protected:
  DispatcherDevice(const std::string &Label, TargetModelType TM, JitT &Jit)
      : Dispatcher(Label, TM), Jit(Jit), CodeCache(Label) {}

  virtual void linkDeviceLibraries(Module &M) = 0;

  JitT &Jit;

private:
  MemoryCache<KernelFunction_t> CodeCache;
};

} // namespace proteus

#endif

#endif // PROTEUS_FRONTEND_DISPATCHER_DEVICE_H
