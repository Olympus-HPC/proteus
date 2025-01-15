// RUN: rm -rf .proteus
// RUN: ./kernel.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./kernel.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-SECOND
#include <climits>
#include <cstdio>

#include "gpu_common.h"

__global__ __attribute__((annotate("jit"))) void kernel() {
  printf("Kernel\n");
}

int main() {
  kernel<<<1, 1>>>();
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// CHECK: Kernel
// CHECK: JitCache hits 0 total 1
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
