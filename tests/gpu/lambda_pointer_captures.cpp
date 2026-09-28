// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_pointer_captures.%ext | %FILECHECK %s --implicit-check-not="[LambdaSpec] Replacing slot 0 "
// RUN: rm -rf "%t.$$.proteus"
// Pointer captures are not auto specialized, scalar captures next to them are.
// clang-format on

#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

template <typename T>
__global__ __attribute__((annotate("jit"))) void kernel(T LB) {
  LB();
}

int main() {
  double *Out = nullptr;
  gpuErrCheck(gpuMallocManaged(&Out, sizeof(double)));
  *Out = 1.0;
  double D = 1.5;
  kernel<<<1, 1>>>(
      proteus::register_lambda([Out, D] __device__() { *Out += D; }));
  gpuErrCheck(gpuDeviceSynchronize());
  printf("Out %.1f\n", *Out);
  gpuErrCheck(gpuFree(Out));

  return 0;
}

// clang-format off
// CHECK: [LambdaSpec] Replacing slot 1 with double 1.500000e+00
// CHECK: Out 2.5
// clang-format on
