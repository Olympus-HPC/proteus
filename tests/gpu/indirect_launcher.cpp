// clang-format off
// RUN: ./indirect_launcher.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./indirect_launcher.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-SECOND
// clang-format on

#include <climits>
#include <cstdio>
#include <iostream>

#include "gpu_common.h"

#define gpuErrCheck(CALL)                                                      \
  {                                                                            \
    gpuError_t err = CALL;                                                     \
    if (err != gpuSuccess) {                                                   \
      printf("ERROR @ %s:%d ->  %s\n", __FILE__, __LINE__,                     \
             gpuGetErrorString(err));                                          \
      abort();                                                                 \
    }                                                                          \
  }
__global__ __attribute__((annotate("jit"))) void kernel() {
  printf("Kernel one\n");
}

__global__ __attribute__((annotate("jit"))) void kernel_two() {
  printf("Kernel two\n");
}

template <typename T> gpuError_t launcher(T kernel_in) {
  return gpuLaunchKernel((const void*)kernel_in, 1, 1, 0, 0, 0);
}

template <typename T> gpuError_t launchertwo(T kernel_in) {
  return launcher(kernel_in);
}

int main() {
  auto indirect = reinterpret_cast<const void*>(&kernel);
  gpuErrCheck(launcher(kernel));
  gpuErrCheck(launcher(kernel_two));
  gpuErrCheck(launchertwo(kernel_two));
  gpuErrCheck(gpuDeviceSynchronize());

  return 0;
}

// CHECK: Kernel
// CHECK: Kernel
// CHECK: JitCache hits 1 total 3
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 2
// CHECK-SECOND: JitStorageCache hits 2 total 2
