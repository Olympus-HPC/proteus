// RUN: ./kernel_args | FileCheck %s
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

__global__ __attribute__((annotate("jit", 1, 2, 3))) void
kernel(int arg1, int arg2, int arg3) {
  printf("Kernel arg %d\n", arg1 + arg2 + arg3);
}

int main() {
  kernel<<<1, 1>>>(3, 2, 1);
  hipErrCheck(hipDeviceSynchronize());
  return 0;
}

// CHECK: JitCache hits 0 total 1
// CHECK: HashValue {{[0-9]+}} num_execs 1
// CHECK: Kernel arg 6