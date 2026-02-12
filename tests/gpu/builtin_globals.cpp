// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_SPECIALIZE_DIMS_RANGE=1 PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/builtin_globals.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_SPECIALIZE_DIMS_RANGE=1 PROTEUS_CACHE_DIR="%t.$$.proteus" %build/builtin_globals.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <climits>
#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

__global__ __attribute__((annotate("jit"))) void kernel() {
  dim3 Idx = threadIdx;
  (void)Idx;
  dim3 Blockidx = blockIdx;
  (void)Blockidx;
  printf("Kernel\n");
}

int main() {
  kernel<<<1, 1>>>();
  gpuErrCheck(gpuDeviceSynchronize());

  return 0;
}

// clang-format off
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [LaunchBoundSpec] MaxThreads 1 MinBlocksPerSM 0
// CHECK: Kernel
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 hits 0 accesses 1
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: [proteus][JitEngineDevice] StorageCache rank 0 hits 0 accesses 1
// CHECK-SECOND: [proteus][JitEngineDevice] StorageCache rank 0 hits 1 accesses 1
