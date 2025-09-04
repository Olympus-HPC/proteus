// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/kernel_launch_exception.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/kernel_launch_exception.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <climits>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <sys/cdefs.h>

#include "gpu_common.h"
#include <proteus/JitInterface.hpp>

__global__ __attribute__((annotate("jit"))) void kernel() {
  printf("Kernel launch with exception\n");
}

int main() {
  proteus::init();

  try {
    if (gpuLaunchKernel((const void *)&kernel, 1, 1, 0, 0, 0) != gpuSuccess)
      throw std::runtime_error("Launch failed");
  } catch (const std::exception &E) {
    std::cerr << "Exception " << E.what() << "\n";
  }
  gpuErrCheck(gpuDeviceSynchronize());

  proteus::finalize();
  return 0;
}

// CHECK: Kernel launch with exception
// CHECK: JitCache hits 0 total 1
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
