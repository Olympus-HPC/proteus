// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_auto_second_arg.%ext | %FILECHECK %s
// RUN: rm -rf "%t.$$.proteus"
// Auto read-only captures of a lambda passed as a later kernel argument.
// clang-format on

#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

template <typename T>
__global__ __attribute__((annotate("jit"))) void kernel(int *Out, T LB) {
  Out[0] = LB();
}

int main() {
  int *Out = nullptr;
  gpuErrCheck(gpuMallocManaged(&Out, sizeof(int)));
  int A = 6;
  int B = 7;
  kernel<<<1, 1>>>(
      Out, proteus::register_lambda([A, B] __device__() { return A * B; }));
  gpuErrCheck(gpuDeviceSynchronize());
  printf("Out %d\n", *Out);
  gpuErrCheck(gpuFree(Out));

  return 0;
}

// clang-format off
// CHECK-DAG: [LambdaSpec] Replacing slot 0 with i32 6
// CHECK-DAG: [LambdaSpec] Replacing slot 1 with i32 7
// CHECK: Out 42
// clang-format on
