// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/indirect_fallthrough.%ext | %FILECHECK %s --check-prefixes=CHECK
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <climits>
#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.hpp>

__global__ void kernel() { printf("Kernel\n"); }

template <typename T> gpuError_t launcher(T KernelIn) {
  return gpuLaunchKernel((const void *)KernelIn, 1, 1, 0, 0, 0);
}

int main() {
  proteus::init();

  gpuErrCheck(launcher(kernel));
  gpuErrCheck(gpuDeviceSynchronize());

  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK: Kernel
