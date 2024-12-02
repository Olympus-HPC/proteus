// clang-format off
// RUN: ./rdc.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./rdc.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-SECOND
// clang-format on

#include "gpu_common.h"
#include <cstdio>

#define gpuErrCheck(CALL)                                                      \
  {                                                                            \
    gpuError_t err = CALL;                                                     \
    if (err != gpuSuccess) {                                                   \
      printf("ERROR @ %s:%d ->  %s\n", __FILE__, __LINE__,                     \
             gpuGetErrorString(err));                                          \
      abort();                                                                 \
    }                                                                          \
  }

// Forward declaration
extern __device__ void device_function(int a);
extern __device__ void device_function2(int a);

__global__ __attribute__((annotate("jit", 1))) void kernel_function(int a) {
  device_function(a);
  device_function2(a);
  printf("Kernel %d\n", a);
};

int main() {
  kernel_function<<<1, 1>>>(1);
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// CHECK: Kernel 1
// CHECK: JitCache hits 0 total 1
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
