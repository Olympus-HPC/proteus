// RUN: rm -rf .proteus
// RUN: ./kernel_host_device_jit.%ext | FileCheck %s --check-prefixes=CHECK
// RUN: rm -rf .proteus
#include <climits>
#include <cstdio>

#include "gpu_common.h"

__global__ __attribute__((annotate("jit"))) void kernel() {
  printf("Kernel\n");
}

template <typename T>
__attribute__((annotate("jit"))) gpuError_t launcher(T kernel_in) {
  return gpuLaunchKernel((const void *)kernel_in, 1, 1, 0, 0, 0);
}

int main() {
  kernel<<<1, 1>>>();
  gpuErrCheck(launcher(kernel));
  gpuErrCheck(gpuDeviceSynchronize());

  return 0;
}

// CHECK: Kernel
// CHECK: Kernel
// CHECK: JitCache hits 1 total 2
// CHECK: HashValue {{[0-9]+}} NumExecs 2 NumHits 1
// CHECK: JitStorageCache hits 0 total 1
