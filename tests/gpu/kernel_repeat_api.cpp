// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/kernel_repeat_api.%ext |  %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/kernel_repeat_api.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on
#include <climits>
#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

__global__ void kernel(int I) {
  proteus::jit_arg(I);
  printf("Kernel i %d\n", I);
}

int main() {
  for (int I = 0; I < 1000; I++)
    kernel<<<1, 1>>>(42);
  gpuErrCheck(gpuDeviceSynchronize());

  return 0;
}

// clang-format off
// CHECK-FIRST: [ArgSpec] Replaced Function _Z6kerneli ArgNo 0 with value i32 42
// CHECK-FIRST: [LaunchBoundSpec] MaxThreads 1 MinBlocksPerSM 0
// CHECK-COUNT-1000: Kernel i 42
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 hits 999 accesses 1000
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1000 NumHits 999
// CHECK-FIRST: [proteus][JitEngineDevice] StorageCache rank 0 hits 0 accesses 1
// CHECK-SECOND: [proteus][JitEngineDevice] StorageCache rank 0 hits 1 accesses 1
