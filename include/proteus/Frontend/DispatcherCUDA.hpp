#ifndef PROTEUS_FRONTEND_DISPATCHER_CUDA_HPP
#define PROTEUS_FRONTEND_DISPATCHER_CUDA_HPP

#if PROTEUS_ENABLE_CUDA

#include "proteus/Frontend/Dispatcher.hpp"

namespace proteus {

class DispatcherCUDA : public Dispatcher {
public:
  static DispatcherCUDA &instance() {
    static DispatcherCUDA D;
    return D;
  }

private:
  JitEngineDeviceCUDA &Jit;
  DispatcherCUDA() : Jit(JitEngineDeviceCUDA::instance()) {}
};

} // namespace proteus

#endif

#endif // PROTEUS_FRONTEND_DISPATCHER_CUDA_HPP