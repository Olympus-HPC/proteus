// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/indirect_launcher.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/indirect_launcher.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <climits>
#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.hpp>

__global__ __attribute__((annotate("jit"))) void kernel() {
  printf("Kernel\n");
}

template <typename T> gpuError_t launcher(T KernelIn) {
  return gpuLaunchKernel((const void *)KernelIn, 1, 1, 0, 0, 0);
}

int main() {
  proteus::init();

  kernel<<<1, 1>>>();
  gpuErrCheck(launcher(kernel));
  gpuErrCheck(gpuDeviceSynchronize());

  proteus::finalize();
  return 0;
}

// CHECK: Kernel
// CHECK: Kernel
// CHECK: JitCache hits 1 total 2
// CHECK: HashValue {{[0-9]+}} NumExecs 2 NumHits 1
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
