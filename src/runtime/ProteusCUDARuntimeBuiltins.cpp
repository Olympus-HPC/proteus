#include "../include/proteus/impl/UtilsCUDA.h"

#include <cuda_runtime.h>

#include <cstdio>

// NOLINTBEGIN(readability-identifier-naming)

// Resolve at runtime CUDA runtime symbols to avoid a dependency on the CUDA
// runtime library for the proteus runtime library, and allow users to link with
// either the static or dynamic CUDA runtime library.

// Implement an internal function for cudaGetSymbolAddress to check the return
// value and report errors for debugging.
static cudaError_t checkCudaGetSymbolAddress(void **DevPtr,
                                             const void *Symbol) {
  cudaError_t Ret = cudaGetSymbolAddress(DevPtr, Symbol);
  proteusCudaErrCheck(Ret);
  return Ret;
}

extern "C" {
// External funciton pointers for the CUDA runtime symbols used by the Proteus
// CUDA runtime.
extern cudaError_t (*__proteus_cudaGetSymbolAddress_ptr)(void **, const void *);
extern cudaError_t (*__proteus_cudaLaunchKernel_ptr)(const void *, dim3, dim3,
                                                     void **, size_t,
                                                     cudaStream_t);
}

// Initialization function to set the function pointers for the CUDA runtime
// symbols used by the Proteus runtime library. This function is called from the
// ProteusPass when detecting a CUDA module.
extern "C" void __proteus_cudart_builtins_init() {
  __proteus_cudaGetSymbolAddress_ptr = checkCudaGetSymbolAddress;
  __proteus_cudaLaunchKernel_ptr = cudaLaunchKernel;
}

// NOLINTEND(readability-identifier-naming)
