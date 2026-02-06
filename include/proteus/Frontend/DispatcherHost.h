#ifndef PROTEUS_FRONTEND_DISPATCHER_HOST_H
#define PROTEUS_FRONTEND_DISPATCHER_HOST_H

#include "proteus/Caching/ObjectCacheRegistry.h"
#include "proteus/CompiledLibrary.h"
#include "proteus/Frontend/Dispatcher.h"
#include "proteus/JitEngineHost.h"

namespace proteus {

class DispatcherHost : public Dispatcher {
public:
  static DispatcherHost &instance() {
    static DispatcherHost D;
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

  DispatchResult launch(void *, LaunchDims, LaunchDims, void *[], uint64_t,
                        void *) override {
    reportFatalError("Host does not support launch");
  }

  StringRef getDeviceArch() const override {
    reportFatalError("Host dispatcher does not implement getDeviceArch");
  }

  void *getFunctionAddress(const std::string &FnName, const HashT &ModuleHash,
                           CompiledLibrary &Library) override {
    HashT FuncHash = hash(FnName, ModuleHash);

    if (void *FuncPtr = CodeCache.lookup(FuncHash))
      return FuncPtr;

    if (!Library.IsLoaded) {
      Jit.loadCompiledLibrary(Library);
      Library.IsLoaded = true;
    }

    void *FuncAddr = Jit.getFunctionAddress(FnName, Library);
    if (!FuncAddr)
      reportFatalError("Failed to find address for function " + FnName);

    CodeCache.insert(FuncHash, FuncAddr, FnName);

    return FuncAddr;
  }

  void registerDynamicLibrary(const HashT &HashValue,
                              const std::string &Path) override {
    auto Buf = MemoryBuffer::getFileAsStream(Path);
    if (!Buf)
      reportFatalError("Failed to read dynamic library: " + Path);
    getObjectCache().store(HashValue,
                           CacheEntry::sharedObject((*Buf)->getMemBufferRef()));
  }

protected:
  DispatcherHost() : Jit(JitEngineHost::instance()) {
    TargetModel = TargetModelType::HOST;
    DispatcherName = "DispatcherHost";
    ObjectCacheRegistry::instance().create(DispatcherName);
  }

  ~DispatcherHost() {
    CodeCache.printStats();
    getObjectCache().printStats();
  }

private:
  JitEngineHost &Jit;
  MemoryCache<void *> CodeCache{"DispatcherHost"};
};

} // namespace proteus

#endif // PROTEUS_FRONTEND_DISPATCHER_HOST_H
