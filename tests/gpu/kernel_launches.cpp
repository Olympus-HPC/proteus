// clang-format off
// RUN: ./kernel_launches.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./kernel_launches.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-SECOND
// clang-format on
#include <climits>
#include <cstdio>

#include "gpu_common.h"

#define gpuErrCheck(CALL)                                                      \
  {                                                                            \
    gpuError_t err = CALL;                                                     \
    if (err != gpuSuccess) {                                                   \
      printf("ERROR @ %s:%d ->  %s\n", __FILE__, __LINE__,                     \
             gpuGetErrorString(err));                                          \
      abort();                                                                 \
    }                                                                          \
  }

__global__ __attribute__((annotate("jit"))) void kernel() {
  printf("Kernel\n");
}

int main() {
  kernel<<<1, 1>>>();
  gpuErrCheck(gpuLaunchKernel((const void *)kernel, 1, 1, nullptr, 0, 0));
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// CHECK: Kernel
// CHECK: JitCache hits 1 total 2
// CHECK: HashValue {{[0-9]+}} NumExecs 2 NumHits 1
// CHECK-FIRST: JitStoredCache hits 0 total 1
// CHECK-SECOND: JitStoredCache hits 1 total 1