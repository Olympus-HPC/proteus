// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT=1 %build/kernel_repeat_api.%ext |  %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/kernel_repeat_api.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on
#include <climits>
#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.hpp>

__global__ void kernel(int I) {
  proteus::jit_arg(I);
  printf("Kernel i %d\n", I);
}

int main() {
  proteus::init();

  for (int I = 0; I < 1000; I++)
    kernel<<<1, 1>>>(42);
  gpuErrCheck(gpuDeviceSynchronize());

  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK-FIRST: [ArgSpec] Replaced Function _Z6kerneli ArgNo 0 with value i32 42
// CHECK-FIRST: [LaunchBoundSpec] MaxThreads 1 MinBlocksPerSM 0
// CHECK-COUNT-1000: Kernel i 42
// CHECK: JitCache hits 999 total 1000
// CHECK: HashValue {{[0-9]+}} NumExecs 1000 NumHits 999
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
