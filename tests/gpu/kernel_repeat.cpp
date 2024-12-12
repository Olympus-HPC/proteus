// RUN: ./kernel_repeat.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./kernel_repeat.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-SECOND
#include <climits>
#include <cstdio>

#include "gpu_common.h"

__global__ __attribute__((annotate("jit", 1))) void kernel(int i) {
  printf("Kernel i %d\n", i);
}

int main() {
  for (int i = 0; i < 1000; i++)
    kernel<<<1, 1>>>(42);
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// CHECK-COUNT-1000: Kernel i 42
// CHECK: JitCache hits 999 total 1000
// CHECK: HashValue {{[0-9]+}} NumExecs 1000 NumHits 999
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
