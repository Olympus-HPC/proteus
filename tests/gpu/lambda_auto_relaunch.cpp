// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization;kernel-trace" %build/lambda_auto_relaunch.%ext | %FILECHECK %s
// RUN: rm -rf "%t.$$.proteus"
// Each distinct auto captured value gets its own specialization, and a
// repeated value reuses it.
// clang-format on

#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

template <typename T>
__global__ __attribute__((annotate("jit"))) void kernel(T LB) {
  LB();
}

int main() {
  int Scale = 10;
  for (int I = 0; I < 6; ++I) {
    int V = I % 3;
    kernel<<<1, 1>>>(proteus::register_lambda(
        [V, Scale] __device__() { printf("V %d Scaled %d\n", V, V * Scale); }));
    gpuErrCheck(gpuDeviceSynchronize());
  }

  return 0;
}

// clang-format off
// CHECK: [LambdaSpec] Replacing slot 0 with i32 0
// CHECK: V 0 Scaled 0
// CHECK: [LambdaSpec] Replacing slot 0 with i32 1
// CHECK: V 1 Scaled 10
// CHECK: [LambdaSpec] Replacing slot 0 with i32 2
// CHECK: V 2 Scaled 20
// CHECK-NOT: [LambdaSpec]
// CHECK: V 0 Scaled 0
// CHECK-NOT: [LambdaSpec]
// CHECK: V 1 Scaled 10
// CHECK-NOT: [LambdaSpec]
// CHECK: V 2 Scaled 20
// CHECK: === Kernel Trace (rank 0) ===
// CHECK: void kernel{{.*}}  rank=0  specializations=3  launches=6
// CHECK: === End Kernel Trace ===
// clang-format on
