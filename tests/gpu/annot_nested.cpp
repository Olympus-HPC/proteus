// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization;kernel-trace" %build/annot_nested.%ext | %FILECHECK %s
// RUN: rm -rf "%t.$$.proteus"
// A directly launched JIT kernel inside a JIT host function specializes on its
// own argument while the enclosing host function reuses its specialization.
// clang-format on

#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

__global__ __attribute__((annotate("jit", 1))) void innerFn(int W) {
  printf("Inner %d\n", W);
}

__attribute__((annotate("jit", 1))) void outerFn(int V, int W) {
  printf("Outer %d\n", V);
  innerFn<<<1, 1>>>(W);
}

int main() {
  for (int W : {10, 20, 30}) {
    outerFn(1, W);
    gpuErrCheck(gpuDeviceSynchronize());
  }

  return 0;
}

// clang-format off
// CHECK: [ArgSpec] Replaced Function _Z7outerFnii ArgNo 0 with value i32 1
// CHECK: Outer 1
// CHECK: [ArgSpec] Replaced Function _Z7innerFni ArgNo 0 with value i32 10
// CHECK: Inner 10
// CHECK: Outer 1
// CHECK: [ArgSpec] Replaced Function _Z7innerFni ArgNo 0 with value i32 20
// CHECK: Inner 20
// CHECK: Outer 1
// CHECK: [ArgSpec] Replaced Function _Z7innerFni ArgNo 0 with value i32 30
// CHECK: Inner 30
// CHECK-DAG: [proteus][JitEngineHost]   outerFn(int, int)  rank=0  specializations=1  launches=3
// CHECK-DAG: [proteus][JitEngineDevice]   innerFn(int)  rank=0  specializations=3  launches=3
// clang-format on
