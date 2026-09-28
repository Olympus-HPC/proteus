// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_aggregate_capture.%ext | %FILECHECK %s --implicit-check-not="with i32 2" --implicit-check-not="with double 3.5"
// RUN: rm -rf "%t.$$.proteus"
// An aggregate captured by value is not auto specialized, the scalar captures
// around it are.
// clang-format on

#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

struct Pair {
  int X;
  double Y;
};

template <typename T>
__global__ __attribute__((annotate("jit"))) void kernel(T LB) {
  LB();
}

int main() {
  int A = 1;
  Pair S{2, 3.5};
  int B = 4;
  kernel<<<1, 1>>>(proteus::register_lambda([A, S, B] __device__() {
    printf("A %d S.X %d S.Y %.1f B %d\n", A, S.X, S.Y, B);
  }));
  gpuErrCheck(gpuDeviceSynchronize());

  return 0;
}

// clang-format off
// CHECK-DAG: [LambdaSpec] Replacing slot 0 with i32 1
// CHECK-DAG: [LambdaSpec] Replacing slot {{[0-9]+}} with i32 4
// CHECK: A 1 S.X 2 S.Y 3.5 B 4
// clang-format on
