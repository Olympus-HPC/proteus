#ifndef PROTEUS_FRONTEND_DISPATCHER_HOST_HIP_H
#define PROTEUS_FRONTEND_DISPATCHER_HOST_HIP_H

#if PROTEUS_ENABLE_HIP

#include "proteus/Frontend/DispatcherHost.h"
#include "proteus/JitEngineDeviceHIP.h"

namespace proteus {

class DispatcherHostHIP : public DispatcherHost {
public:
  static DispatcherHostHIP &instance() {
    static DispatcherHostHIP D;
    return D;
  }

  StringRef getDeviceArch() const override {
    return JitEngineDeviceHIP::instance().getDeviceArch();
  }

private:
  DispatcherHostHIP() { TargetModel = TargetModelType::HOST_HIP; }
};

} // namespace proteus

#endif

#endif // PROTEUS_FRONTEND_DISPATCHER_HOST_HIP_H
