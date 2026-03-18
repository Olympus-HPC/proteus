#include "proteus/impl/Frontend/CppJitFuncAttribute.h"

#include "proteus/Error.h"

#if PROTEUS_ENABLE_CUDA
#include "proteus/impl/UtilsCUDA.h"
#endif

#if PROTEUS_ENABLE_HIP
#include "proteus/impl/UtilsHIP.h"
#endif

namespace proteus {

void setFuncAttribute(TargetModelType TargetModel, void *KernelFunc,
                      CppJitFuncAttribute Attr, int Value) {
  if (Value < 0)
    reportFatalError("Function attribute value must be non-negative");

  switch (TargetModel) {
#if PROTEUS_ENABLE_CUDA
  case TargetModelType::CUDA:
    switch (Attr) {
    case CppJitFuncAttribute::MaxDynamicSharedMemorySize:
      proteusCuErrCheck(cuFuncSetAttribute(
          reinterpret_cast<CUfunction>(KernelFunc),
          CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, Value));
      return;
    }
#else
  case TargetModelType::CUDA:
    reportFatalError("CUDA function attributes require PROTEUS_ENABLE_CUDA");
#endif
#if PROTEUS_ENABLE_HIP
  case TargetModelType::HIP:
    switch (Attr) {
    case CppJitFuncAttribute::MaxDynamicSharedMemorySize:
      proteusHipErrCheck(hipFuncSetAttribute(
          reinterpret_cast<hipFunction_t>(KernelFunc),
          hipFuncAttributeMaxDynamicSharedMemorySize, Value));
      return;
    }
#else
  case TargetModelType::HIP:
    reportFatalError("HIP function attributes require PROTEUS_ENABLE_HIP");
#endif
  case TargetModelType::HOST:
  case TargetModelType::HOST_CUDA:
  case TargetModelType::HOST_HIP:
    reportFatalError(
        "Host launchers do not support direct function attributes");
  }

  reportFatalError("Unsupported function attribute for target model");
}

} // namespace proteus
