// clang-format off
// RUN: rm -rf .proteus
// RUN: PROTEUS_TRACE_OUTPUT=1 ./mix_attr_api.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./mix_attr_api.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf .proteus
// clang-format on
#include <climits>
#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.hpp>

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
// CHECK-FIRST: [LaunchBoundSpec] GridSize 1 BlockSize 1
// CHECK: Kernel Attr 1 2.000000 3.000000
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9kernelAPIifd ArgNo 0 with value i32 1
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9kernelAPIifd ArgNo 1 with value float 2.000000e+00
// CHECK-FIRST: [ArgSpec] Replaced Function _Z9kernelAPIifd ArgNo 2 with value double 3.000000e+00
// CHECK-FIRST: [LaunchBoundSpec] GridSize 1 BlockSize 1
// CHECK: Kernel API 1 2.000000 3.000000
// CHECK: JitCache hits 0 total 2
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 2
// CHECK-SECOND: JitStorageCache hits 2 total 2
