// RUN: rm -rf .proteus
// RUN: ./kernels_gvar.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./kernels_gvar.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf .proteus
#include <climits>
#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.hpp>

__device__ int Gvar = 23;

__global__ __attribute__((annotate("jit"))) void kernel() {
  Gvar++;
  printf("Kernel gvar %d addr %p\n", Gvar, &Gvar);
}

__global__ __attribute__((annotate("jit"))) void kernel2() {
  Gvar++;
  printf("Kernel2 gvar %d addr %p\n", Gvar, &Gvar);
}

__global__ void kernel3() {
  Gvar++;
  printf("Kernel3 gvar %d addr %p\n", Gvar, &Gvar);
}

int main() {
  proteus::init();

  kernel<<<1, 1>>>();
  gpuErrCheck(gpuDeviceSynchronize());
  kernel2<<<1, 1>>>();
  gpuErrCheck(gpuDeviceSynchronize());
  kernel3<<<1, 1>>>();
  gpuErrCheck(gpuDeviceSynchronize());

  proteus::finalize();
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
