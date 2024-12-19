#if ENABLE_CUDA
#include <cuda_runtime.h>
#define gpuError_t cudaError_t
#define gpuSuccess cudaSuccess
#define gpuGetErrorString cudaGetErrorString
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuMallocManaged cudaMallocManaged
#define gpuFree cudaFree
#define gpuLaunchKernel cudaLaunchKernel
#elif ENABLE_HIP
#include <hip/hip_runtime.h>
#define gpuError_t hipError_t
#define gpuSuccess hipSuccess
#define gpuGetErrorString hipGetErrorString
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuMallocManaged hipMallocManaged
#define gpuFree hipFree
#define gpuLaunchKernel hipLaunchKernel
#else
#error "Must provide ENABLE_HIP or ENABLE_CUDA"
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

struct kernel_body {
  __device__ void operator()() { printf("Kernel"); }
};

const kernel_body my_kernel_body{};
