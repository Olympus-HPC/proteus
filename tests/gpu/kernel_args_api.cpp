// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT=1 %build/kernel_args_api.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/kernel_args_api.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on
#include <climits>
#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.hpp>

__global__ void kernel(int Arg1, int Arg2, int Arg3) {
  proteus::jit_arg(Arg1);
  proteus::jit_arg(Arg2);
  proteus::jit_arg(Arg3);
  printf("Kernel arg %d\n", Arg1 + Arg2 + Arg3);
}

int main() {
  proteus::init();

  kernel<<<1, 1>>>(3, 2, 1);
  gpuErrCheck(gpuDeviceSynchronize());

  proteus::finalize();
  return 0;
}
// clang-format off
// CHECK-FIRST: [ArgSpec] Replaced Function _Z6kerneliii ArgNo 0 with value i32 3
// CHECK-FIRST: [ArgSpec] Replaced Function _Z6kerneliii ArgNo 1 with value i32 2
// CHECK-FIRST: [ArgSpec] Replaced Function _Z6kerneliii ArgNo 2 with value i32 1
// CHECK-FIRST: [LaunchBoundSpec] MaxThreads 1 MinBlocksPerSM 0
// CHECK: Kernel arg 6
// CHECK: [proteus][JitEngineDevice] MemoryCache procuid 0 hits 0 accesses 1
// CHECK: [proteus][JitEngineDevice] MemoryCache procuid 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: [proteus][JitEngineDevice] StorageCache procuid 0 hits 0 accesses 1
// CHECK-SECOND: [proteus][JitEngineDevice] StorageCache procuid 0 hits 1 accesses 1
