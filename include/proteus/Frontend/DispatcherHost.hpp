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

  void compile(std::unique_ptr<Module> M) override {
    Jit.compileOnly(std::move(M));
  }

  DispatchResult launch(StringRef KernelName, LaunchDims GridDim,
                        LaunchDims BlockDim, ArrayRef<void *> KernelArgs,
                        uint64_t ShmemSize, void *Stream) override {
    PROTEUS_FATAL_ERROR("Host does not support launch");
  }

protected:
  void *getFunctionAddress(StringRef FnName) override {
    return Jit.getFunctionAddress(FnName);
  }

private:
  JitEngineHost &Jit;
  DispatcherHost() : Jit(JitEngineHost::instance()) {}
};

} // namespace proteus

#endif // PROTEUS_FRONTEND_DISPATCHER_HOST_HPP