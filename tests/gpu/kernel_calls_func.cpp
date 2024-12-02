// clang-format off
// RUN: ./kernel_calls_func.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./kernel_calls_func.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-SECOND
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
__device__ void device_function(int a);

__global__ __attribute__((annotate("jit"))) void kernel_function(int a) {
  device_function(a);
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
