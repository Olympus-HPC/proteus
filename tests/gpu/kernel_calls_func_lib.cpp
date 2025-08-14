// clang-format off
// RUN: rm -rf .proteus
// RUN: PROTEUS_TRACE_OUTPUT=1 ./kernel_calls_func_lib.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./kernel_calls_func_lib.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf .proteus
// clang-format on

#include "gpu_common.h"
#include <cstdio>
#include <proteus/JitInterface.hpp>

// Forward declaration
extern __device__ void deviceFunction(int A);

__global__ __attribute__((annotate("jit", 1))) void kernelFunction(int A) {
  deviceFunction(A);
  printf("Kernel with lib %d\n", A);
};

int main() {
  proteus::init();

  kernelFunction<<<1, 1>>>(1);
  gpuErrCheck(gpuDeviceSynchronize());

  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK-FIRST: [ArgSpec] Replaced Function _Z14kernelFunctioni ArgNo 0 with value i32 1
// CHECK-FIRST: [LaunchBoundSpec] BlockSize 1
// CHECK: device_func 1
// CHECK: Kernel with lib 1
// CHECK: JitCache hits 0 total 1
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
