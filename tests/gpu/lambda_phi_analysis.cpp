// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization;cache-stats" %build/lambda_phi_analysis.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="cache-stats" %build/lambda_phi_analysis.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdint>
#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

template <typename Lambda>
__global__ __attribute__((annotate("jit"))) void
kernel_phi(Lambda Body, Lambda, bool CollapseToKernelArgTwo) {
  helper_slot(&Body, &Body, true, CollapseToKernelArgTwo);
}

template <typename F> __device__ void invoke(F *Fn) { (*Fn)(); }

template <typename F> struct Box {
  F *Fn;
};

template <typename F>
__device__ void helper_slot(F *Opt1, F *Opt2, bool Cond, bool IndexTwo) {
  F *Slot = nullptr;
  if (!IndexTwo) {
    Box<F> B{Opt2};
    if (Cond)
      Slot = Opt1;
    else
      Slot = B.Fn;
  } else {
    Box<F> B{Opt1};
    if (Cond)
      Slot = Opt2;
    else
      Slot = B.Fn;
  }
  invoke(Slot);
}

template <typename F> __device__ void helper_box(F *Fn) {
  Box<F> B{Fn};
  invoke(B.Fn);
}

auto getTestLambda(int V) {
  return proteus::register_lambda(
      [=, X = proteus::jit_variable(V)] __host__ __device__ {
        printf("case phi %d\n", X);
      });
}

// Case 3: multiple stores to the same temporary slot, only one dominating the
// actual wrapper call on the executed path.

static void phi_case(int V1, int V2, bool CollapseToKernelArgTwo) {
  // branch should always resovle to 55
  if (CollapseToKernelArgTwo)
    kernel_phi<<<1, 1>>>(getTestLambda(V1), getTestLambda(V2),
                         CollapseToKernelArgTwo);
  // branch should always resolve to 77
  else
    kernel_phi<<<1, 1>>>(getTestLambda(V2), getTestLambda(V1),
                         CollapseToKernelArgTwo);
  gpuErrCheck(gpuDeviceSynchronize());
}

int main() {
  int V1 = 55;
  int V2 = 77;
  // resolve to 77
  phi_case(V1, V2, false);
  // resolve to 55
  phi_case(V2, V1, false);

  // resolve to 55
  phi_case(V1, V2, true);
  // resolve to 77
  phi_case(V2, V1, true);

  return 0;
}

// clang-format off
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 77
// CHECK: case phi 77
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 55
// CHECK: case phi 55
// CHECK: case phi 55
// CHECK: case phi 77
// CHECK-FIRST: [proteus][JitEngineDevice] MemoryCache rank 0 hits 2 accesses 4
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 2 NumHits 1
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 2 NumHits 1
// CHECK-FIRST: [proteus][JitEngineDevice] StorageCache rank 0 hits 0 accesses 2
// CHECK-SECOND: [proteus][JitEngineDevice] StorageCache rank 0 hits 2 accesses 2
// clang-format on
