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

#include <llvm/Support/MemoryBuffer.h>

#include <mutex>

namespace proteus {

template <typename JitT> class DispatcherDevice : public Dispatcher {
public:
  using KernelFunction_t = typename DeviceTraits<JitT>::KernelFunction_t;

  std::unique_ptr<MemoryBuffer>
  compileModule(Module &M, const CodeGenerationConfig &CGConfig) override {
    TIMESCOPE(DispatcherDevice, compileModule);

    linkDeviceLibraries(M);
    optimizeModule(M, CGConfig);
    return codegenModule(M, CGConfig);
  }

  void optimizeModule(Module &M,
                      const CodeGenerationConfig &CGConfig) override {
    TIMESCOPE(DispatcherDevice, optimizeModule);

    if (JitT::optimizesBeforeCodegen(CGConfig.codeGenOption())) {
      proteus::optimizeIR(M, Jit.getDeviceArch(),
                          OptimizationPipelineConfig(CGConfig));
      return;
    }

    if (CGConfig.codeGenOption() == CodegenOption::RTC)
      warnOptimizationConfigIgnoredByRTC(CGConfig);
  }

  std::unique_ptr<MemoryBuffer>
  codegenModule(Module &M, const CodeGenerationConfig &CGConfig) override {
    TIMESCOPE(DispatcherDevice, codegenModule);

    auto ObjBuf = Jit.codegenObject(M, Jit.GlobalLinkedBinaries, CGConfig);
    if (!ObjBuf)
      reportFatalError("Expected non-null object library");

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

  void *lookupFunction(const KernelName &Name,
                       const HashT &ModuleHash) override {
    HashT HashValue = hash(Name.mangled(), ModuleHash);
    return CodeCache.lookup(HashValue);
  }

  void *insertFunction(const KernelName &Name, const HashT &ModuleHash,
                       CompiledLibrary &Library) override {
    TIMESCOPE(DispatcherDevice, insertFunction);
    HashT HashValue = hash(Name.mangled(), ModuleHash);

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
        Name.mangled(), Library.ObjectModule->getBufferStart(),
        Library.RelinkGlobalsByCopy, VarNameToGlobalInfo);
    Library.IsLoaded = true;

    CodeCache.insert(HashValue, KernelFunc, Name.base());

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
  // Skipping optimization on the RTC path leaves the runtime compiler in charge
  // of it, which silently drops every user-configured optimization setting.
  // Warn once per process so the setting does not appear to take effect.
  static void
  warnOptimizationConfigIgnoredByRTC(const CodeGenerationConfig &CGConfig) {
    // '3' and 3 are the PROTEUS_OPT_LEVEL and PROTEUS_CODEGEN_OPT_LEVEL
    // defaults set by CodeGenerationConfig.
    const bool UsesDefaults =
        !CGConfig.optPipeline() && CGConfig.optLevel() == '3' &&
        CGConfig.codeGenOptLevel() == 3 && getJITPassPluginConfigs().empty();
    if (UsesDefaults)
      return;

    static std::once_flag WarnOnce;
    std::call_once(WarnOnce, [] {
      Logger::outs("proteus")
          << "Warning: RTC codegen optimizes internally, so Proteus ignores "
             "PROTEUS_OPT_PIPELINE, PROTEUS_OPT_LEVEL, "
             "PROTEUS_CODEGEN_OPT_LEVEL and JIT pass plugins, use "
             "PROTEUS_CODEGEN=serial or PROTEUS_CODEGEN=parallel to apply "
             "them\n";
    });
  }

  MemoryCache<KernelFunction_t> CodeCache;
};

} // namespace proteus

#endif

#endif // PROTEUS_FRONTEND_DISPATCHER_DEVICE_H
