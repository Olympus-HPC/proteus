// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_auto_readonly.%ext | %FILECHECK %s
// RUN: rm -rf "%t.$$.proteus"
// Scalar captures that the lambda only reads are specialized without
// jit_variable.
// clang-format on

#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

template <typename T>
__global__ __attribute__((annotate("jit"))) void kernel(T LB) {
  LB();
}

int main() {
  int I = 7;
  double D = 2.5;
  float F = 1.5f;
  bool B = true;
  long L = 3;
  kernel<<<1, 1>>>(proteus::register_lambda([I, D, F, B, L] __device__() {
    printf("I %d D %.2f F %.2f B %d L %ld\n", I, D, F, B, L);
  }));
  gpuErrCheck(gpuDeviceSynchronize());

  return 0;
}

// clang-format off
// CHECK-DAG: [LambdaSpec] Replacing slot 0 with i32 7
// CHECK-DAG: [LambdaSpec] Replacing slot 1 with double 2.500000e+00
// CHECK-DAG: [LambdaSpec] Replacing slot 2 with float 1.500000e+00
// CHECK-DAG: [LambdaSpec] Replacing slot 3 with i8 1
// CHECK-DAG: [LambdaSpec] Replacing slot 4 with i64 3
// CHECK: I 7 D 2.50 F 1.50 B 1 L 3
// clang-format on
