// RUN: rm -rf .proteus
// RUN: ./kernel_repeat.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./kernel_repeat.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf .proteus
#include <climits>
#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.hpp>

__global__ __attribute__((annotate("jit", 1))) void kernel(int I) {
  printf("Kernel i %d\n", I);
}

int main() {
  proteus::init();

  for (int I = 0; I < 1000; I++)
    kernel<<<1, 1>>>(42);
  gpuErrCheck(gpuDeviceSynchronize());

  proteus::finalize();
  return 0;
}

// CHECK-COUNT-1000: Kernel i 42
// CHECK: JitCache hits 999 total 1000
// CHECK: HashValue {{[0-9]+}} NumExecs 1000 NumHits 999
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
