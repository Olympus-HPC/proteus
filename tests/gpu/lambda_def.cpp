// clang-format off
// RUN: rm -rf .proteus
// RUN: PROTEUS_TRACE_OUTPUT=1 ./lambda_def.%ext |  %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./lambda_def.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf .proteus
// clang-format on

#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.hpp>

template <typename T>
__global__ __attribute__((annotate("jit"))) void kernel(T LB) {
  LB();
}

template <typename T> void run(T &&LB) {
  proteus::register_lambda(LB);
  kernel<<<1, 1>>>(LB);
  gpuErrCheck(gpuDeviceSynchronize());
}

int main() {
  proteus::init();

  int A = 42;
  auto Lambda =
      [=, A = proteus::jit_variable(A)] __device__()
          __attribute__((annotate("jit"))) { printf("Lambda A %d\n", A); };
  run(Lambda);
  run(Lambda);
  run(Lambda);

  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 42
// CHECK-COUNT-3: Lambda A 42
// CHECK: JitCache hits 2 total 3
// CHECK: HashValue {{[0-9]+}} NumExecs 3 NumHits 2
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
