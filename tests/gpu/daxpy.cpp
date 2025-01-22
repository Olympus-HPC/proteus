// RUN: ./daxpy.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./daxpy.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-SECOND
#include <cstddef>
#include <cstdlib>
#include <iostream>

#include "gpu_common.h"

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

int main(int argc, char **argv) {
  int N = 1024;
  double *x;
  double *y;

  gpuErrCheck(gpuMallocManaged(&x, sizeof(double) * N));
  gpuErrCheck(gpuMallocManaged(&y, sizeof(double) * N));

  for (std::size_t i{0}; i < N; i++) {
    x[i] = 0.31414 * i;
    y[i] = 0.0;
  }

  std::cout << y[10] << std::endl;
  daxpy(6.2, x, y, N);
  gpuErrCheck(gpuDeviceSynchronize());
  std::cout << y[10] << std::endl;
  daxpy(6.2, x, y, N);
  gpuErrCheck(gpuDeviceSynchronize());
  std::cout << y[10] << std::endl;

  gpuErrCheck(gpuFree(x));
  gpuErrCheck(gpuFree(y));
}

// CHECK: 0
// CHECK: 19944.1
// CHECK: 39888.2
// CHECK: JitCache hits 1 total 2
// CHECK: HashValue {{[0-9]+}} NumExecs 2 NumHits 1
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
