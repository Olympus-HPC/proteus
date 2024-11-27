// clang-format off
// RUN: ./indirect_launcher_multi_arg.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./indirect_launcher_multi_arg.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-SECOND
// clang-format on

#include <climits>
#include <cstdio>

#include "gpu_common.h"

__global__ __attribute__((annotate("jit", 1)))void kernel(int arg) {
  printf("Kernel one; arg = %d\n", arg);
}

__global__ __attribute__((annotate("jit", 1))) void kernel_two(int arg) {
  printf("Kernel two; arg = %d\n", arg);
}

gpuError_t launcher(const void *kernel_in, int a) {
  void *args[] = {&a};
  return gpuLaunchKernel(kernel_in, 1, 1, args, 0, 0);
}

int main() {
  gpuErrCheck(launcher((const void *)kernel, 42));
  gpuErrCheck(launcher((const void *)kernel_two, 24));
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// CHECK: Kernel one; arg = 42
// CHECK: Kernel two; arg = 24
// CHECK: JitCache hits 0 total 2
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 2
// CHECK-SECOND: JitStorageCache hits 2 total 2
