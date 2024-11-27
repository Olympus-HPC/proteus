// clang-format off
// RUN: ./indirect_launcher_arg.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./indirect_launcher_arg.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-SECOND
// clang-format on

#include <climits>
#include <cstdio>

#include "gpu_common.h"

__global__ __attribute__((annotate("jit", 1))) void kernel(int a) {
  printf("Kernel %d\n", a);
}

template <typename T> gpuError_t launcher(T kernel_in, int a) {
  void *args[] = {&a};
  return gpuLaunchKernel((const void *)kernel_in, 1, 1, args, 0, 0);
}

int main() {
  kernel<<<1, 1>>>(42);
  gpuErrCheck(launcher(kernel, 24));
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// CHECK: Kernel 42
// CHECK: Kernel 24
// CHECK: JitCache hits 0 total 2
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 2
// CHECK-SECOND: JitStorageCache hits 2 total 2
