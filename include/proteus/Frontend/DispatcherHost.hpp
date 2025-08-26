#ifndef PROTEUS_FRONTEND_DISPATCHER_HOST_HPP
#define PROTEUS_FRONTEND_DISPATCHER_HOST_HPP

#include "proteus/Frontend/Dispatcher.hpp"
#include "proteus/JitEngineHost.hpp"

namespace proteus {

class DispatcherHost : public Dispatcher {
public:
  static DispatcherHost &instance() {
    static DispatcherHost D;
    return D;
  }

  std::unique_ptr<MemoryBuffer>
  compile(std::unique_ptr<LLVMContext> Ctx, std::unique_ptr<Module> M,
          [[maybe_unused]] HashT ModuleHash) override {
    // ModuleHash is unused since we do not implement an object cache for host
    // JIT.
    Jit.compileOnly(std::move(Ctx), std::move(M));
    // TODO: Host compilation is managed by ORC lazy JIT and does not support
    // synchronous object creation.
    return nullptr;
  }

  std::unique_ptr<MemoryBuffer> lookupObjectModule(HashT) override {
    // Host JIT does not implement object caching.
    return nullptr;
  }

  DispatchResult launch(void *, LaunchDims, LaunchDims, ArrayRef<void *>,
                        uint64_t, void *) override {
    PROTEUS_FATAL_ERROR("Host does not support launch");
  }

  StringRef getTargetArch() const override {
    PROTEUS_FATAL_ERROR("Host dispatcher does not implement getTargetArch");
  }

  void *getFunctionAddress(StringRef FnName,
                           std::optional<MemoryBufferRef>) override {
    // ObjectModule is unused, the ORC JIT singleton has a single global module.
    void *FuncAddr = Jit.getFunctionAddress(FnName);
    if (!FuncAddr)
      PROTEUS_FATAL_ERROR("Failed to find address for function " + FnName);

    return FuncAddr;
  }

private:
  // TODO: The JitEngineHost is a singleton and consolidates all compiled IR in
  // a single object layer. This creates name collision for same named functions
  // (duplicate definitions) though they are in different Jit modules.
  // Reconsider singletons for both the JitEngineHost and the Dispatcher.
  JitEngineHost &Jit;
  DispatcherHost() : Jit(JitEngineHost::instance()) {
    TargetModel = TargetModelType::HOST;
  }
};

} // namespace proteus

#endif // PROTEUS_FRONTEND_DISPATCHER_HOST_HPP
