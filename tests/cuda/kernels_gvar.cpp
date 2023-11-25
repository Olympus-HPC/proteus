// RUN: ./kernels_gvar.%ext | FileCheck %s
#include <climits>
#include <cstdio>
#include <cuda_runtime.h>

__device__ int gvar = 23;

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
  gvar++;
  printf("Kernel gvar %d addr %p\n", gvar, &gvar);
}

__global__ __attribute__((annotate("jit"))) void kernel2() {
  gvar++;
  printf("Kernel2 gvar %d addr %p\n", gvar, &gvar);
}

__global__ void kernel3() {
  gvar++;
  printf("Kernel3 gvar %d addr %p\n", gvar, &gvar);
}

int main() {
  kernel<<<1, 1>>>();
  cudaErrCheck(cudaDeviceSynchronize());
  kernel2<<<1, 1>>>();
  cudaErrCheck(cudaDeviceSynchronize());
  kernel3<<<1, 1>>>();
  cudaErrCheck(cudaDeviceSynchronize());
  return 0;
}

// CHECK: Kernel gvar 24 addr [[ADDR:[a-z0-9]+]]
// CHECK: Kernel2 gvar 25 addr [[ADDR]]
// CHECK: Kernel3 gvar 26 addr [[ADDR]]
// CHECK: JitCache hits 0 total 2
// CHECK: HashValue {{[0-9]+}} num_execs 1
// CHECK: HashValue {{[0-9]+}} num_execs 1
