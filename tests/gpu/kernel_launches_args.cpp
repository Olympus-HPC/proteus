// clang-format off
// RUN: rm -rf .proteus
// RUN: ./kernel_launches_args.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./kernel_launches_args.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf .proteus
// clang-format on
#include <climits>
#include <cstdio>

#include "gpu_common.h"

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
  gpuErrCheck(gpuLaunchKernel((const void *)kernel, 1, 1, args, 0, 0));
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// CHECK: Kernel 24 44
// CHECK: JitCache hits 1 total 2
// CHECK: HashValue {{[0-9]+}} NumExecs 2 NumHits 1
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
