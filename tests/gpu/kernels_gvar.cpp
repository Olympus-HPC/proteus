// RUN: rm -rf .proteus
// RUN: ./kernels_gvar.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./kernels_gvar.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf .proteus
#include <climits>
#include <cstdio>

#include "gpu_common.h"

__device__ int gvar = 23;

__global__ __attribute__((annotate("jit"))) void kernel() {
  gvar++;
  printf("Kernel gvar %d addr %p\n", gvar, &gvar);
}

__global__ __attribute__((annotate("jit"))) void kernel2() {
  gvar++;
  printf("Kernel2 gvar %d addr %p\n", gvar, &gvar);
}

__global__ void kernel3() {
  gvar++;
  printf("Kernel3 gvar %d addr %p\n", gvar, &gvar);
}

int main() {
  kernel<<<1, 1>>>();
  gpuErrCheck(gpuDeviceSynchronize());
  kernel2<<<1, 1>>>();
  gpuErrCheck(gpuDeviceSynchronize());
  kernel3<<<1, 1>>>();
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// CHECK: Kernel gvar 24 addr [[ADDR:[a-z0-9]+]]
// CHECK: Kernel2 gvar 25 addr [[ADDR]]
// CHECK: Kernel3 gvar 26 addr [[ADDR]]
// CHECK: JitCache hits 0 total 2
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 2
// CHECK-SECOND: JitStorageCache hits 2 total 2
