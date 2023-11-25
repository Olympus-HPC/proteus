// RUN: ./kernel_cache.%ext | FileCheck %s
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
  for (int i = 0; i < 10; ++i) {
    kernel<<<1, 1>>>();
    cudaErrCheck(cudaDeviceSynchronize());
  }
  return 0;
}

// CHECK-COUNT-10: Kernel
// CHECK-NOT: Kernel
// CHECK: JitCache hits 9 total 10
// CHECK: HashValue {{[0-9]+}} num_execs 10
