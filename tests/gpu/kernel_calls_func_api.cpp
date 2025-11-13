// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/kernel_calls_func.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/kernel_calls_func.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
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

// clang-format off
// CHECK: device_func 1
// CHECK: Kernel 1
// CHECK: [proteus][JitEngineDevice] MemoryCache procuid 0 hits 0 accesses 1
// CHECK: [proteus][JitEngineDevice] MemoryCache procuid 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: [proteus][JitEngineDevice] StorageCache procuid 0 hits 0 accesses 1
// CHECK-SECOND: [proteus][JitEngineDevice] StorageCache procuid 0 hits 1 accesses 1
