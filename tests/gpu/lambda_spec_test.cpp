// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT=1 PROTEUS_DEBUG_OUTPUT=1 %build/lambda_spec_test.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT=1 PROTEUS_DEBUG_OUTPUT=1 %build/lambda_spec_test.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <climits>
#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

__device__ void printInt(int I) { printf("Integer = %d\n", I); }

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
  int zero = 0;
  int one = 1;
  int two = 2;

  auto zero_lambda = [=, c = proteus::jit_variable(zero)] __device__()
                         __attribute__((annotate("jit"))) { printInt(c); };

  auto one_lambda = [=, c = proteus::jit_variable(one)] __device__()
                        __attribute__((annotate("jit"))) { printInt(c); };

  auto two_lambda = [=, c = proteus::jit_variable(two)] __device__()
                        __attribute__((annotate("jit"))) { printInt(c); };

  run(zero_lambda);
  run(one_lambda);
  run(two_lambda);
  proteus::finalize();

  return 0;
}

// clang-format off
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 0
// CHECK: Integer = 0
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 1
// CHECK: Integer = 1
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 2
// CHECK: Integer = 2
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0 FnName _Z6kernelIZ4mainEUlvE1_EvT_
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0 FnName _Z6kernelIZ4mainEUlvE_EvT_
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0 FnName _Z6kernelIZ4mainEUlvE0_EvT_
// CHECK-FIRST: [proteus][JitEngineDevice] StorageCache rank 0 hits 0 accesses 3
// CHECK-SECOND: [proteus][JitEngineDevice] StorageCache rank 0 hits 3 accesses 3
