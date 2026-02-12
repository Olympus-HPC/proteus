// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/kernel_cache.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/kernel_cache.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// Test buildFromConfig parsing with explicit PROTEUS_OBJECT_CACHE_CHAIN.
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_OBJECT_CACHE_CHAIN="storage" PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/kernel_cache.%ext 2>&1 | %FILECHECK %s --check-prefix=CHECK-CONFIG
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <climits>
#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

__global__ __attribute__((annotate("jit"))) void kernel() {
  printf("Kernel\n");
}

int main() {
  for (int I = 0; I < 10; ++I) {
    kernel<<<1, 1>>>();
    gpuErrCheck(gpuDeviceSynchronize());
  }

  return 0;
}

// clang-format off
// CHECK-COUNT-10: Kernel
// CHECK-NOT: Kernel
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 hits 9 accesses 10
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 10 NumHits 9
// CHECK-FIRST: [proteus][JitEngineDevice] StorageCache rank 0 hits 0 accesses 1
// CHECK-SECOND: [proteus][JitEngineDevice] StorageCache rank 0 hits 1 accesses 1
// CHECK-CONFIG: [ObjectCacheChain] Added cache level: storage
// CHECK-CONFIG: [ObjectCacheChain] Chain for JitEngineDevice: Storage
// clang-format on
