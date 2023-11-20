// RUN: ./kernel | FileCheck %s
#include <climits>
#include <cstdio>
#include <hip/hip_runtime.h>

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
  printf("Kernel\n");
}

int main() {
  kernel<<<1, 1>>>();
  hipErrCheck(hipDeviceSynchronize());
  return 0;
}

// CHECK: JitCache hits 0 total 1
// CHECK: HashValue {{[0-9]+}} num_execs 1
// CHECK: Kernel