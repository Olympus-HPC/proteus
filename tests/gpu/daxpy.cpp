// RUN: rm -rf .proteus
// RUN: ./daxpy.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./daxpy.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf .proteus

#include <cstddef>
#include <cstdlib>
#include <iostream>

#include "gpu_common.h"
#include <proteus/JitInterface.hpp>

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
  void *Args[] = {&A, &X, &Y, &N};
  cudaLaunchKernel((const void *)(daxpyImpl), dim3(GridSize), dim3(256), Args,
                   0, 0);
#else
#error Must provide PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA
#endif
}

int main(int argc, char **argv) {
  proteus::init();

  int N = 1024;
  double *X;
  double *Y;

  gpuErrCheck(gpuMallocManaged(&X, sizeof(double) * N));
  gpuErrCheck(gpuMallocManaged(&Y, sizeof(double) * N));

  for (std::size_t I{0}; I < N; I++) {
    X[I] = 0.31414 * I;
    Y[I] = 0.0;
  }

  std::cout << Y[10] << std::endl;
  daxpy(6.2, X, Y, N);
  gpuErrCheck(gpuDeviceSynchronize());
  std::cout << Y[10] << std::endl;
  daxpy(6.2, X, Y, N);
  gpuErrCheck(gpuDeviceSynchronize());
  std::cout << Y[10] << std::endl;

  gpuErrCheck(gpuFree(X));
  gpuErrCheck(gpuFree(Y));
}

// CHECK: 0
// CHECK: 19944.1
// CHECK: 39888.2
// CHECK: JitCache hits 1 total 2
// CHECK: HashValue {{[0-9]+}} NumExecs 2 NumHits 1
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
