// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT=1 PROTEUS_TRACE_OUTPUT="specialization;cache-stats" %build/lambda_phi_launcher.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT=1 PROTEUS_TRACE_OUTPUT="specialization;cache-stats" %build/lambda_phi_launcher.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <climits>
#include <cstdio>
#include <tuple>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

__device__ void printInt(int I) { printf("Integer = %d\n", I); }

template <typename T>
__global__ __attribute__((annotate("jit"))) void kernel(T LB, int a) {
  LB(a);
}

template<typename L1, typename L2, typename L3>
struct Abstraction {
  L1 lambda_1;
  L2 lambda_2;
  L3 lambda_3;
  Abstraction(L1&& l1, L2&& l2, L3&& l3) : lambda_1(l1), lambda_2(l2), lambda_3(l3){};

  __host__ __device__ auto operator()(int a) {
    // if (a > 4)
    //   lambda_1();
    // else
    //   lambda_2();
    auto copy = lambda_1;
    copy();
    lambda_3();
  }
};

// Macro can be invoked function style, either within a Function where the arguments passed
// to jit_variable are function parameters, or Constants as below
auto declareLambda(int rc1, int rc2) {
  return PROTEUS_REGISTER_LAMBDA([=, C = proteus::jit_variable(rc1), D = 5,
          C2 = proteus::jit_variable(rc2)]() __attribute__((annotate("jit"))) {
    printInt(C);
    printInt(C2);
    printInt(D);
  });
}

template <typename T> void run(T &&LB) {
  int a = 5;
  kernel<<<1, 1>>>(LB, a);
  gpuErrCheck(gpuDeviceSynchronize());
}

int main() {
  int Zero = 0;
  int One = 1;
  int Two = 2;
  // auto RUNTIME_CONSTANT_TUPLE_ONE = {Zero, One};
  auto ZeroLambda = declareLambda(Zero, One);
  // proteus::LamdbaFunctorWrapper<RUNTIME_CONSTANT_TUPLE_ONE>
  auto OneLambda = declareLambda(One, Two);

  auto TwoLambda = declareLambda(Two, Zero);

  Abstraction A (std::move(ZeroLambda), std::move(OneLambda), std::move(TwoLambda));

  run(A);

  return 0;
}

// clang-format off
// CHECK-FIRST: [LambdaSpec] Replacing slot 2 with i32 1
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 0
// CHECK-FIRST: [LambdaSpec] Replacing slot 2 with i32 0
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 2
// CHECK: Integer = 0
// CHECK: Integer = 1
// CHECK: Integer = 5
// CHECK: Integer = 2
// CHECK: Integer = 0
// CHECK: Integer = 5
// CHECK-FIRST: [proteus][JitEngineDevice] MemoryCache rank 0 hits 0 accesses 1
// CHECK-COUNT-1: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: [proteus][JitEngineDevice] StorageCache rank 0 hits 0 accesses 1
// CHECK-SECOND: [proteus][JitEngineDevice] StorageCache rank 0 hits 1 accesses 1
