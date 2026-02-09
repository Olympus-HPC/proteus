#include "proteus/Frontend/Dispatcher.h"

#include "proteus/Error.h"
#include "proteus/impl/Frontend/DispatcherHost.h"
#if PROTEUS_ENABLE_HIP
#include "proteus/impl/Frontend/DispatcherHIP.h"
#include "proteus/impl/Frontend/DispatcherHostHIP.h"
#endif
#if PROTEUS_ENABLE_CUDA
#include "proteus/impl/Frontend/DispatcherCUDA.h"
#include "proteus/impl/Frontend/DispatcherHostCUDA.h"
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

Dispatcher::Dispatcher(const std::string &Name, TargetModelType TM)
    : TargetModel(TM), Cache(Name) {}

Dispatcher::~Dispatcher() = default;

} // namespace proteus
