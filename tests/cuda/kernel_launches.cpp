// RUN: ./kernel_launches.%ext | FileCheck %s
#include <climits>
#include <cstdio>
#include <cuda_runtime.h>

#define cudaErrCheck(CALL)                                                     \
  {                                                                            \
    cudaError_t err = CALL;                                                    \
    if (err != cudaSuccess) {                                                  \
      printf("ERROR @ %s:%d ->  %s\n", __FILE__, __LINE__,                     \
             cudaGetErrorString(err));                                         \
      abort();                                                                 \
    }                                                                          \
  }

__global__ __attribute__((annotate("jit"))) void kernel() {
  printf("Kernel\n");
}

int main() {
  kernel<<<1, 1>>>();
  cudaErrCheck(cudaLaunchKernel((const void *)kernel, 1, 1, nullptr, 0, 0));
  cudaErrCheck(cudaDeviceSynchronize());
  return 0;
}

// CHECK: Kernel
// CHECK: JitCache hits 1 total 2
// CHECK: HashValue {{[0-9]+}} num_execs 2
