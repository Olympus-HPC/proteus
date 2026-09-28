// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_auto_two_callsites.%ext | %FILECHECK %s
// RUN: rm -rf "%t.$$.proteus"
// One kernel calls two instances of the same lambda with different captured
// values and a second lambda; each callsite is specialized for its own values.
// clang-format on

#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

template <typename L1, typename L2, typename L3>
__global__ __attribute__((annotate("jit"))) void kernel(L1 First, L2 Second,
                                                        L3 Third) {
  First();
  Second();
  Third();
}

static auto makeLambda(int V, double S) {
  return [V, S] __device__() { printf("V %d S %.1f\n", V, S); };
}

int main() {
  int K = 7;
  kernel<<<1, 1>>>(
      proteus::register_lambda(makeLambda(1, 0.5)),
      proteus::register_lambda(makeLambda(2, 1.5)),
      proteus::register_lambda([K] __device__() { printf("K %d\n", K); }));
  gpuErrCheck(gpuDeviceSynchronize());

  return 0;
}

// clang-format off
// CHECK-DAG: [LambdaSpec] Replacing slot 0 with i32 1
// CHECK-DAG: [LambdaSpec] Replacing slot 1 with double 5.000000e-01
// CHECK-DAG: [LambdaSpec] Replacing slot 0 with i32 2
// CHECK-DAG: [LambdaSpec] Replacing slot 1 with double 1.500000e+00
// CHECK-DAG: [LambdaSpec] Replacing slot 0 with i32 7
// CHECK: V 1 S 0.5
// CHECK: V 2 S 1.5
// CHECK: K 7
// clang-format on
