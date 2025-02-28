// clang-format off
// RUN: rm -rf .proteus
// RUN: ./kernel_launch_exception.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./kernel_launch_exception.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf .proteus
// clang-format on

#include <climits>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <sys/cdefs.h>

#include "gpu_common.h"

__global__ __attribute__((annotate("jit"))) void kernel() {
  printf("Kernel launch with exception\n");
}

int main() {
  try {
    if (gpuLaunchKernel((const void *)&kernel, 1, 1, 0, 0, 0) != gpuSuccess)
      throw std::runtime_error("Launch failed");
  } catch (const std::exception &E) {
    std::cerr << "Exception " << E.what() << "\n";
  }
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// CHECK: Kernel launch with exception
// CHECK: JitCache hits 0 total 1
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
