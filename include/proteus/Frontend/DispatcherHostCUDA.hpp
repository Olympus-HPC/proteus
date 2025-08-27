#ifndef PROTEUS_FRONTEND_DISPATCHER_HOST_CUDA_HPP
#define PROTEUS_FRONTEND_DISPATCHER_HOST_CUDA_HPP

#if PROTEUS_ENABLE_CUDA

#include "proteus/Frontend/DispatcherHost.hpp"
#include "proteus/JitEngineDeviceCUDA.hpp"

namespace proteus {

class DispatcherHostCUDA : public DispatcherHost {
public:
  static DispatcherHostCUDA &instance() {
    static DispatcherHostCUDA D;
    return D;
  }

  StringRef getDeviceArch() const override {
    return JitEngineDeviceCUDA::instance().getDeviceArch();
  }

private:
  DispatcherHostCUDA() { TargetModel = TargetModelType::HOST_CUDA; }
};

} // namespace proteus

#endif

#endif // PROTEUS_FRONTEND_DISPATCHER_HOST_CUDA_HPP
