// RUN: rm -rf .proteus
// RUN: ./enable_disable.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./enable_disable.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf .proteus
#include <climits>
#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.hpp>

__global__ __attribute__((annotate("jit"))) void kernel() {
  printf("Kernel\n");
}

int main() {
  proteus::init();

  proteus::enable();
  kernel<<<1, 1>>>();
  gpuErrCheck(gpuDeviceSynchronize());
  proteus::disable();
  kernel<<<1, 1>>>();
  gpuErrCheck(gpuDeviceSynchronize());
  proteus::enable();
  kernel<<<1, 1>>>();
  gpuErrCheck(gpuDeviceSynchronize());

  proteus::finalize();
  return 0;
}

// CHECK: Kernel
// CHECK: Kernel
// CHECK: Kernel
// CHECK: JitCache hits 0 total 0
// CHECK: JitCache hits 1 total 2
// CHECK: HashValue {{[0-9]+}} NumExecs 2 NumHits 1
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
