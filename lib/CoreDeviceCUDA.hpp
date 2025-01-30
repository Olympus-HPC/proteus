#ifndef PROTEUS_CORE_CUDA_HPP
#define PROTEUS_CORE_CUDA_HPP

#include <cuda_runtime.h>

namespace proteus {

static inline cudaError_t launchKernelDirect(void *KernelFunc, dim3 GridDim,
                                             dim3 BlockDim, void **KernelArgs,
                                             uint64_t ShmemSize,
                                             CUstream Stream) {
  return cudaLaunchKernel(KernelFunc, GridDim, BlockDim, KernelArgs, ShmemSize,
                          Stream);
}

} // namespace proteus

#endif
