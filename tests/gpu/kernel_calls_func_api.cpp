// clang-format off
// RUN: rm -rf .proteus
// RUN: ./kernel_calls_func.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./kernel_calls_func.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf .proteus
// clang-format on

#include "gpu_common.h"
#include <cstdio>
#include <proteus/JitInterface.hpp>

// Forward declaration
extern __device__ void deviceFunction(int A);

__global__ void kernelFunction(int A, int B) {
  proteus::jit_arg(A);
  proteus::jit_arg(B);
  deviceFunction(A);
  printf("Kernel %d\n", A);
};

int main() {
  proteus::init();

  kernelFunction<<<1, 1>>>(1, 2);
  gpuErrCheck(gpuDeviceSynchronize());

  proteus::finalize();
  return 0;
}

// CHECK: device_func 1
// CHECK: Kernel 1
// CHECK: JitCache hits 0 total 1
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
