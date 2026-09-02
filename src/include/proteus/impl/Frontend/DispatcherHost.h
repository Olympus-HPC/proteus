#ifndef PROTEUS_FRONTEND_DISPATCHER_HOST_H
#define PROTEUS_FRONTEND_DISPATCHER_HOST_H

#include "proteus/Frontend/Dispatcher.h"
#include "proteus/TimeTracing.h"
#include "proteus/impl/Caching/MemoryCache.h"
#include "proteus/impl/Caching/ObjectCacheChain.h"
#include "proteus/impl/CompiledLibrary.h"
#include "proteus/impl/Config.h"
#include "proteus/impl/JitEngineHost.h"

namespace proteus {

class DispatcherHost : public Dispatcher {
public:
  static DispatcherHost &instance() {
    static DispatcherHost D{"DispatcherHost", JitEngineHost::instance()};
    return D;
  }

  DispatcherHost(const std::string &Label, JitEngineHost &Jit)
      : Dispatcher(Label, TargetModelType::HOST), Jit(Jit), CodeCache(Label) {}

  std::unique_ptr<MemoryBuffer>
  compileModule(Module &M, const CompileOptions &Opts) override {
    TIMESCOPE(DispatcherHost, compileModule);
    const CodeGenerationConfig &CGConfig =
        Opts.CGConfig ? *Opts.CGConfig : Config::get().getCGConfig();
    return Jit.compileOnly(M, CGConfig, Opts.DisableIROpt);
  }

  DispatchResult launch(void *, LaunchDims, LaunchDims, void *[], uint64_t,
                        void *) override {
    reportFatalError("Host does not support launch");
  }

  StringRef getDeviceArch() const override {
    reportFatalError("Host dispatcher does not implement getDeviceArch");
  }

  void *lookupFunction(const std::string &FnName,
                       const HashT &ModuleHash) override {
    HashT FuncHash = hash(FnName, ModuleHash);
    return CodeCache.lookup(FuncHash);
  }

  void *loadFunctionAddress(const std::string &FnName, const HashT &ModuleHash,
                            CompiledLibrary &Library,
                            const std::string &TraceName = "") override {
    TIMESCOPE(DispatcherHost, loadFunctionAddress);
    HashT FuncHash = hash(FnName, ModuleHash);

    if (!Library.IsLoaded) {
      Jit.loadCompiledLibrary(Library);
      Library.IsLoaded = true;
    }

    void *FuncAddr = Jit.getFunctionAddress(FnName, Library);
    if (!FuncAddr)
      reportFatalError("Failed to find address for function " + FnName);

    CodeCache.insert(FuncHash, FuncAddr,
                     TraceName.empty() ? FnName : TraceName);

    return FuncAddr;
  }

  void registerDynamicLibrary(const HashT &HashValue,
                              const std::string &Path) override {
    if (!ObjectCache)
      return;
    auto Buf = MemoryBuffer::getFileAsStream(Path);
    if (!Buf)
      reportFatalError("Failed to read dynamic library: " + Path);
    ObjectCache->store(HashValue,
                       CacheEntry::sharedObject((*Buf)->getMemBufferRef()));
  }

  ~DispatcherHost() {
    if (Config::get().traceCacheStats())
      CodeCache.printStats();
    CodeCache.printKernelTrace();
    printObjectCacheStats();
  }

protected:
  explicit DispatcherHost(const std::string &Label, JitEngineHost &Jit,
                          TargetModelType TM)
      : Dispatcher(Label, TM), Jit(Jit), CodeCache(Label) {}

private:
  JitEngineHost &Jit;
  MemoryCache<void *> CodeCache;
};

} // namespace proteus

#endif // PROTEUS_FRONTEND_DISPATCHER_HOST_H
