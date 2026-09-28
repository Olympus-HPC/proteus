// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_field_pointer_escape.%ext | %FILECHECK %s --implicit-check-not="[LambdaSpec] Replacing slot 1 "
// RUN: rm -rf "%t.$$.proteus"
// A capture whose address escapes the lambda is not specialized.
// clang-format on

#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

__device__ __attribute__((noinline)) void observe(const int *Ptr) {
  printf("observe %d\n", *Ptr);
}

template <typename T>
__global__ __attribute__((annotate("jit"))) void kernel(T LB) {
  LB();
}

int main() {
  int A = 1;
  int B = 2;
  int C = 3;
  kernel<<<1, 1>>>(proteus::register_lambda([A, B, C] __device__() {
    observe(&B);
    printf("A %d B %d C %d\n", A, B, C);
  }));
  gpuErrCheck(gpuDeviceSynchronize());

  return 0;
}

// clang-format off
// CHECK-DAG: [LambdaSpec] Replacing slot 0 with i32 1
// CHECK-DAG: [LambdaSpec] Replacing slot 2 with i32 3
// CHECK: observe 2
// CHECK: A 1 B 2 C 3
// clang-format on
