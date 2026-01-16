// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT=1 %build/mix_attr_api.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/mix_attr_api.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on
#include <climits>
#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

__global__ __attribute__((annotate("jit", 1, 2, 3))) void
kernelAttr(int A, float B, double C) {
  printf("Kernel Attr %d %f %lf\n", A, B, C);
}

__global__ void kernelAPI(int A, float B, double C) {
  proteus::jit_arg(A);
  proteus::jit_arg(B);
  proteus::jit_arg(C);
  printf("Kernel API %d %f %lf\n", A, B, C);
}

int main() {
  proteus::init();

  kernelAttr<<<1, 1>>>(1, 2.0f, 3.0);
  gpuErrCheck(gpuDeviceSynchronize());
  kernelAPI<<<1, 1>>>(1, 2.0f, 3.0);
  gpuErrCheck(gpuDeviceSynchronize());

  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK-FIRST: [ArgSpec] Replaced Function _Z10kernelAttrifd ArgNo 0 with value i32 1
// CHECK-FIRST: [ArgSpec] Replaced Function _Z10kernelAttrifd ArgNo 1 with value float 2.000000e+00
// CHECK-FIRST: [ArgSpec] Replaced Function _Z10kernelAttrifd ArgNo 2 with value double 3.000000e+00
// CHECK-FIRST: [LaunchBoundSpec] MaxThreads 1 MinBlocksPerSM 0
// CHECK: Kernel Attr 1 2.000000 3.000000
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9kernelAPIifd ArgNo 0 with value i32 1
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9kernelAPIifd ArgNo 1 with value float 2.000000e+00
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9kernelAPIifd ArgNo 2 with value double 3.000000e+00
// CHECK-FIRST: [LaunchBoundSpec] MaxThreads 1 MinBlocksPerSM 0
// CHECK: Kernel API 1 2.000000 3.000000
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 hits 0 accesses 2
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: [proteus][JitEngineDevice] StorageCache rank 0 hits 0 accesses 2
// CHECK-SECOND: [proteus][JitEngineDevice] StorageCache rank 0 hits 2 accesses 2
