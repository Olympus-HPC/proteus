// RUN: rm -rf .proteus
// RUN: ./kernel_args.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./kernel_args.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf .proteus
#include <climits>
#include <cstdio>

#include "gpu_common.h"

__global__ __attribute__((annotate("jit", 1, 2, 3))) void
kernel(int arg1, int arg2, int arg3) {
  printf("Kernel arg %d\n", arg1 + arg2 + arg3);
}

int main() {
  kernel<<<1, 1>>>(3, 2, 1);
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// CHECK: Kernel arg 6
// CHECK: JitCache hits 0 total 1
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
