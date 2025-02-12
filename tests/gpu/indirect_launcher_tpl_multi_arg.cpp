// clang-format off
// RUN: rm -rf .proteus
// RUN: ./indirect_launcher_tpl_multi_arg.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./indirect_launcher_tpl_multi_arg.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf .proteus
// clang-format on

#include <climits>
#include <cstdio>
#include <iostream>

#include "gpu_common.h"

__global__ __attribute__((annotate("jit", 1)))void kernel(int arg) {
  printf("Kernel one; arg = %d\n", arg);
}

__global__ __attribute__((annotate("jit", 1))) void kernel_two(int arg) {
  printf("Kernel two; arg = %d\n", arg);
}

template <typename T> gpuError_t launcher(T kernel_in, int a) {
  void *args[] = {&a};
  return gpuLaunchKernel((const void*)kernel_in, 1, 1, args, 0, 0);
}

int main() {
  auto indirect = reinterpret_cast<const void*>(&kernel);
  gpuErrCheck(launcher(kernel, 42));
  gpuErrCheck(launcher(kernel_two, 24));
  gpuErrCheck(gpuDeviceSynchronize());

  return 0;
}

// CHECK: Kernel one; arg = 42
// CHECK: Kernel two; arg = 24
// CHECK: JitCache hits 0 total 2
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 2
// CHECK-SECOND: JitStorageCache hits 2 total 2
