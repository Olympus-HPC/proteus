// RUN: ./kernel_cache | FileCheck %s
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
  for (int i = 0; i < 10; ++i) {
    kernel<<<1, 1>>>();
    hipErrCheck(hipDeviceSynchronize());
  }
  return 0;
}

// CHECK: JitCache hits 9 total 10
// CHECK: HashValue {{[0-9]+}} num_execs 10
// CHECK-COUNT-10: Kernel
// CHECK-NOT: Kernel
