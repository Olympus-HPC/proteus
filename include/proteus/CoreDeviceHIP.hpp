#ifndef CORE_HIP_HPP
#define CORE_HIP_HPP

#include <hip/hip_runtime.h>

namespace proteus {

inline hipError_t launchKernelDirect(void *KernelFunc, dim3 GridDim,
                                     dim3 BlockDim, void **KernelArgs,
                                     uint64_t ShmemSize, hipStream_t Stream) {
  return hipLaunchKernel(KernelFunc, GridDim, BlockDim, KernelArgs, ShmemSize,
                         Stream);
}

} // namespace proteus

#endif
