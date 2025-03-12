#include <cstddef>
#include <cstdlib>
#include <iostream>

#include "../../../gpu/gpu_common.h"

__global__ __attribute__((annotate("jit", 4), noinline)) void
daxpy_impl(double a, double *x, double *y, int N) {
  std::size_t i = blockIdx.x * 256 + threadIdx.x;
  if (i < N) {
    for (int j = 0; j < N; ++j)
      y[i] += x[i] * a;
  }
}

void daxpy(double a, double *x, double *y, int N) {
  const std::size_t grid_size = (((N) + (256) - 1) / (256));
#if PROTEUS_ENABLE_HIP
  hipLaunchKernelGGL((daxpy_impl), dim3(grid_size), dim3(256), 0, 0, a, x, y,
                     N);
#elif PROTEUS_ENABLE_CUDA
  void *args[] = {&a, &x, &y, &N};
  cudaLaunchKernel((const void *)(daxpy_impl), dim3(grid_size), dim3(256), args,
                   0, 0);
#else
#error Must provide PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA
#endif
}
