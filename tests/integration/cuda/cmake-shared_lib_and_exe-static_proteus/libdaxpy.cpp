#include <cstddef>
#include <cstdlib>
#include <iostream>

#include "../../../gpu/gpu_common.h"

__global__ __attribute__((annotate("jit", 4), noinline)) void
daxpyImpl(double A, double *X, double *Y, int N) {
  std::size_t I = blockIdx.x * 256 + threadIdx.x;
  if (I < N) {
    for (int J = 0; J < N; ++J)
      Y[I] += X[I] * A;
  }
}

void daxpy(double A, double *X, double *Y, int N) {
  const std::size_t GridSize = (((N) + (256) - 1) / (256));
#if PROTEUS_ENABLE_HIP
  hipLaunchKernelGGL((daxpyImpl), dim3(GridSize), dim3(256), 0, 0, A, X, Y, N);
#elif PROTEUS_ENABLE_CUDA
  void *args[] = {&A, &X, &Y, &N};
  cudaLaunchKernel((const void *)(daxpyImpl), dim3(GridSize), dim3(256), args,
                   0, 0);
#else
#error Must provide PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA
#endif
}
