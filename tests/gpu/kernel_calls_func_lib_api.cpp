// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT=1 %build/kernel_calls_func_lib_api.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/kernel_calls_func_lib_api.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include "gpu_common.h"
#include <cstdio>
#include <proteus/JitInterface.hpp>

// Forward declaration
extern __device__ void deviceFunction(int A);

__global__ void kernelFunction(int A) {
  proteus::jit_arg(A);
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
// CHECK-FIRST: [LaunchBoundSpec] MaxThreads 1 MinBlocksPerSM 0
// CHECK: device_func 1
// CHECK: Kernel with lib 1
// CHECK: [proteus][JitEngineDevice] MemoryCache procuid 0 hits 0 accesses 1
// CHECK: [proteus][JitEngineDevice] MemoryCache procuid 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: [proteus][JitEngineDevice] StorageCache procuid 0 hits 0 accesses 1
// CHECK-SECOND: [proteus][JitEngineDevice] StorageCache procuid 0 hits 1 accesses 1
