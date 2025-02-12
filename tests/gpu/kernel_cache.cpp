// RUN: rm -rf .proteus
// RUN: ./kernel_cache.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./kernel_cache.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf .proteus
#include <climits>
#include <cstdio>

#include "gpu_common.h"

__global__ __attribute__((annotate("jit"))) void kernel() {
  printf("Kernel\n");
}

int main() {
  for (int i = 0; i < 10; ++i) {
    kernel<<<1, 1>>>();
    gpuErrCheck(gpuDeviceSynchronize());
  }
  return 0;
}

// CHECK-COUNT-10: Kernel
// CHECK-NOT: Kernel
// CHECK: JitCache hits 9 total 10
// CHECK: HashValue {{[0-9]+}} NumExecs 10 NumHits 9
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
