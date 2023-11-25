// RUN: ./kernel_launches_args.%ext | FileCheck %s
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

__global__ __attribute__((annotate("jit"))) void kernel(int a, int b) {
  a += 1;
  b += 2;
  printf("Kernel %d %d\n", a, b);
}

int main() {
  int a = 23;
  int b = 42;
  kernel<<<1, 1>>>(a, b);
  void *args[] = {&a, &b};
  hipErrCheck(hipLaunchKernel((const void *)kernel, 1, 1, args, 0, 0));
  hipErrCheck(hipDeviceSynchronize());
  return 0;
}

// CHECK: Kernel 24 44
// CHECK: JitCache hits 1 total 2
// CHECK: HashValue {{[0-9]+}} num_execs 2
