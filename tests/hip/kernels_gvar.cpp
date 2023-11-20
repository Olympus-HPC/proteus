// RUN: ./kernels_gvar | FileCheck %s
#include <climits>
#include <cstdio>
#include <hip/hip_runtime.h>

__device__ int gvar = 23;

#define hipErrCheck(CALL)                                                      \
  {                                                                            \
    hipError_t err = CALL;                                                     \
    if (err != hipSuccess) {                                                   \
      printf("ERROR @ %s:%d ->  %s\n", __FILE__, __LINE__,                     \
             hipGetErrorString(err));                                          \
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
  hipErrCheck(hipDeviceSynchronize());
  kernel2<<<1, 1>>>();
  hipErrCheck(hipDeviceSynchronize());
  kernel3<<<1, 1>>>();
  hipErrCheck(hipDeviceSynchronize());
  return 0;
}

// CHECK: JitCache hits 0 total 2
// CHECK: HashValue {{[0-9]+}} num_execs 1
// CHECK: HashValue {{[0-9]+}} num_execs 1
// CHECK: Kernel gvar 24 addr [[ADDR:[a-z0-9]+]]
// CHECK: Kernel2 gvar 25 addr [[ADDR]]
// CHECK: Kernel3 gvar 26 addr [[ADDR]]
