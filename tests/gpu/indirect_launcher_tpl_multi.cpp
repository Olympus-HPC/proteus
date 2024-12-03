// clang-format off
// RUN: ./indirect_launcher_tpl_multi.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./indirect_launcher_tpl_multi.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-SECOND
// clang-format on

#include <climits>
#include <cstdio>
#include <iostream>

#include "gpu_common.h"

__global__ __attribute__((annotate("jit"))) void kernel() {
  printf("Kernel one\n");
}

__global__ __attribute__((annotate("jit"))) void kernel_two() {
  printf("Kernel two\n");
}

template <typename T> gpuError_t launcher(T kernel_in) {
  return gpuLaunchKernel((const void*)kernel_in, 1, 1, 0, 0, 0);
}

int main() {
  gpuErrCheck(launcher(kernel));
  gpuErrCheck(launcher(kernel_two));
  gpuErrCheck(gpuDeviceSynchronize());

  return 0;
}

// CHECK: Kernel
// CHECK: Kernel two
// CHECK: JitCache hits 0 total 2
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 2
// CHECK-SECOND: JitStorageCache hits 2 total 2