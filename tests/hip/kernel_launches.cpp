// RUN: ./kernel_launches.%ext | FileCheck %s
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
  hipErrCheck(hipLaunchKernel((const void *)kernel, 1, 1, nullptr, 0, 0));
  hipErrCheck(hipDeviceSynchronize());
  return 0;
}

// CHECK: Kernel
// CHECK: JitCache hits 1 total 2
// CHECK: HashValue {{[0-9]+}} num_execs 2
