#ifndef PROTEUS_FRONTEND_DISPATCHER_HOST_CUDA_H
#define PROTEUS_FRONTEND_DISPATCHER_HOST_CUDA_H

#if PROTEUS_ENABLE_CUDA

#include "proteus/impl/Frontend/DispatcherHost.h"
#include "proteus/impl/JitEngineDeviceCUDA.h"

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

#endif // PROTEUS_FRONTEND_DISPATCHER_HOST_CUDA_H
