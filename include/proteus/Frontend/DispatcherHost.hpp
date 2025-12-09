#ifndef PROTEUS_FRONTEND_DISPATCHER_HOST_HPP
#define PROTEUS_FRONTEND_DISPATCHER_HOST_HPP

#include "proteus/Caching/ObjectCacheChain.hpp"
#include "proteus/CompiledLibrary.hpp"
#include "proteus/Frontend/Dispatcher.hpp"
#include "proteus/JitEngineHost.hpp"

namespace proteus {

class DispatcherHost : public Dispatcher {
public:
  static DispatcherHost &instance() {
    static DispatcherHost D;
    return D;
  }

  std::unique_ptr<MemoryBuffer> compile(std::unique_ptr<LLVMContext> Ctx,
                                        std::unique_ptr<Module> Mod,
                                        HashT ModuleHash,
                                        bool DisableIROpt = false) override {
    // This is necessary to ensure Ctx outlives M. Setting [[maybe_unused]] can
    // trigger a lifetime bug.
    auto CtxOwner = std::move(Ctx);
    auto ModOwner = std::move(Mod);
    std::unique_ptr<MemoryBuffer> ObjectModule =
        Jit.compileOnly(*ModOwner, DisableIROpt);
    if (!ObjectModule)
      PROTEUS_FATAL_ERROR("Expected non-null object library");

    ObjectCache.store(
        ModuleHash, CacheEntry::staticObject(ObjectModule->getMemBufferRef()));

    return ObjectModule;
  }

  std::unique_ptr<CompiledLibrary>
  lookupCompiledLibrary(HashT ModuleHash) override {
    return ObjectCache.lookup(ModuleHash);
  }

  DispatchResult launch(void *, LaunchDims, LaunchDims, ArrayRef<void *>,
                        uint64_t, void *) override {
    PROTEUS_FATAL_ERROR("Host does not support launch");
  }

  StringRef getDeviceArch() const override {
    PROTEUS_FATAL_ERROR("Host dispatcher does not implement getDeviceArch");
  }

  void *getFunctionAddress(StringRef FnName, HashT ModuleHash,
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
      PROTEUS_FATAL_ERROR("Failed to find address for function " + FnName);

    CodeCache.insert(FuncHash, FuncAddr, FnName);

    return FuncAddr;
  }

  void registerDynamicLibrary(HashT HashValue,
                              const SmallString<128> &Path) override {
    auto Buf = MemoryBuffer::getFileAsStream(Path);
    if (!Buf)
      PROTEUS_FATAL_ERROR("Failed to read dynamic library: " + Path);
    ObjectCache.store(HashValue,
                      CacheEntry::sharedObject((*Buf)->getMemBufferRef()));
  }

protected:
  DispatcherHost() : Jit(JitEngineHost::instance()) {
    TargetModel = TargetModelType::HOST;
  }

  ~DispatcherHost() {
    CodeCache.printStats();
    ObjectCache.printStats();
  }

private:
  JitEngineHost &Jit;
  MemoryCache<void *> CodeCache{"DispatcherHost"};
  ObjectCacheChain ObjectCache{"DispatcherHost"};
};

} // namespace proteus

#endif // PROTEUS_FRONTEND_DISPATCHER_HOST_HPP
