#include "proteus/Frontend/Dispatcher.hpp"
#include "proteus/Error.h"

#include "proteus/Frontend/DispatcherHost.hpp"

#if PROTEUS_ENABLE_HIP
#include "proteus/Frontend/DispatcherHIP.hpp"
#endif

#if PROTEUS_ENABLE_CUDA
#include "proteus/Frontend/DispatcherCUDA.hpp"
#endif

namespace proteus {
Dispatcher &Dispatcher::getDispatcher(TargetModelType Model) {
  switch (Model) {
  case TargetModelType::HOST:
    return DispatcherHost::instance();
  case TargetModelType::HIP:
#if PROTEUS_ENABLE_HIP
    return DispatcherHIP::instance();
#else
    PROTEUS_FATAL_ERROR("HIP support is not enabled");
#endif
  case TargetModelType::CUDA:
#if PROTEUS_ENABLE_CUDA
    return DispatcherCUDA::instance();
#else
    PROTEUS_FATAL_ERROR("CUDA support is not enabled");
#endif
  default:
    PROTEUS_FATAL_ERROR("Unsupport model type");
  }
}

} // namespace proteus
