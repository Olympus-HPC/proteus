// RUN: ./kernel_args.%ext | FileCheck %s
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

__global__ __attribute__((annotate("jit", 1, 2, 3))) void
kernel(int arg1, int arg2, int arg3) {
  printf("Kernel arg %d\n", arg1 + arg2 + arg3);
}

int main() {
  kernel<<<1, 1>>>(3, 2, 1);
  cudaErrCheck(cudaDeviceSynchronize());
  return 0;
}

// CHECK: Kernel arg 6
// CHECK: JitCache hits 0 total 1
// CHECK: HashValue {{[0-9]+}} num_execs 1