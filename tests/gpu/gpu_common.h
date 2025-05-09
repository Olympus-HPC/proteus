#if PROTEUS_ENABLE_CUDA
#include <cuda_runtime.h>
#define gpuError_t cudaError_t
#define gpuStream_t cudaStream_t
#define gpuSuccess cudaSuccess
#define gpuGetErrorString cudaGetErrorString
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuMallocManaged cudaMallocManaged
#define gpuFree cudaFree
#define gpuLaunchKernel cudaLaunchKernel
#define gpuMemcpyFromSymbol cudaMemcpyFromSymbol
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuStreamCreate cudaStreamCreate
#define gpuStreamSynchronize cudaStreamSynchronize
#define gpuStreamDestroy cudaStreamDestroy
#elif PROTEUS_ENABLE_HIP
#include <hip/hip_runtime.h>
#define gpuError_t hipError_t
#define gpuStream_t hipStream_t
#define gpuSuccess hipSuccess
#define gpuGetErrorString hipGetErrorString
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuMallocManaged hipMallocManaged
#define gpuFree hipFree
#define gpuLaunchKernel hipLaunchKernel
#define gpuMemcpyFromSymbol hipMemcpyFromSymbol
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuStreamCreate hipStreamCreate
#define gpuStreamSynchronize hipStreamSynchronize
#define gpuStreamDestroy hipStreamDestroy
#else
#error "Must provide PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA"
#endif

#define gpuErrCheck(CALL)                                                      \
  {                                                                            \
    gpuError_t err = CALL;                                                     \
    if (err != gpuSuccess) {                                                   \
      printf("ERROR @ %s:%d ->  %s\n", __FILE__, __LINE__,                     \
             gpuGetErrorString(err));                                          \
      abort();                                                                 \
    }                                                                          \
  }
