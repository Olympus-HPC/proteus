// clang-format off
// RUN: ./indirect_fallthrough.%ext | FileCheck %s --check-prefixes=CHECK
// clang-format on

#include <climits>
#include <cstdio>

#include "gpu_common.h"

__global__ void kernel() {
  printf("Kernel\n");
}

template <typename T> gpuError_t launcher(T kernel_in) {
  return gpuLaunchKernel((const void *)kernel_in, 1, 1, 0, 0, 0);
}

int main() {
  gpuErrCheck(launcher(kernel));
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// CHECK: Kernel
