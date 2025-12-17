#include "proteus/Frontend/Dispatcher.hpp"

#include "proteus/Error.h"
#include "proteus/Frontend/DispatcherHost.hpp"
#if PROTEUS_ENABLE_HIP
#include "proteus/Frontend/DispatcherHIP.hpp"
#include "proteus/Frontend/DispatcherHostHIP.hpp"
#endif
#if PROTEUS_ENABLE_CUDA
#include "proteus/Frontend/DispatcherCUDA.hpp"
#include "proteus/Frontend/DispatcherHostCUDA.hpp"
#endif

namespace proteus {

namespace {

Dispatcher &getHostHIPDispatcher() {
#if PROTEUS_ENABLE_HIP
  return DispatcherHostHIP::instance();
#else
  reportFatalError("HIP support is not enabled");
#endif
}

Dispatcher &getHostCUDADispatcher() {
#if PROTEUS_ENABLE_CUDA
  return DispatcherHostCUDA::instance();
#else
  reportFatalError("CUDA support is not enabled");
#endif
}

Dispatcher &getHostDispatcher() { return DispatcherHost::instance(); }

Dispatcher &getHIPDispatcher() {
#if PROTEUS_ENABLE_HIP
  return DispatcherHIP::instance();
#else
  reportFatalError("HIP support is not enabled");
#endif
}

Dispatcher &getCUDADispatcher() {
#if PROTEUS_ENABLE_CUDA
  return DispatcherCUDA::instance();
#else
  reportFatalError("CUDA support is not enabled");
#endif
}
} // anonymous namespace

Dispatcher &Dispatcher::getDispatcher(TargetModelType TargetModel) {
  switch (TargetModel) {
  case TargetModelType::HOST_HIP:
    return getHostHIPDispatcher();
  case TargetModelType::HOST_CUDA:
    return getHostCUDADispatcher();
  case TargetModelType::HOST:
    return getHostDispatcher();
  case TargetModelType::HIP:
    return getHIPDispatcher();
  case TargetModelType::CUDA:
    return getCUDADispatcher();
  default:
    reportFatalError("Unsupported model type");
  }
}

Dispatcher::~Dispatcher() = default;

} // namespace proteus
