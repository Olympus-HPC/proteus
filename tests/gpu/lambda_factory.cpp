// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT=1 PROTEUS_TRACE_OUTPUT="specialization;cache-stats" %build/lambda_factory.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT=1 PROTEUS_TRACE_OUTPUT="specialization;cache-stats" %build/lambda_factory.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
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


auto declareLambda(int rc1, int rc2) {
  return [=, C = proteus::jit_variable(rc1), D = 5,
          C2 = proteus::jit_variable(rc2)]() __attribute__((annotate("jit"))) {
    printInt(C);
    printInt(C2);
    printInt(D);
  };
}

auto declareLambdaThree(int rc1, int rc2, int rc3) {
  return [=, C = proteus::jit_variable(rc1), D = 4,
          C2 = proteus::jit_variable(rc2), rc3 = proteus::jit_variable(rc3)]()
             __attribute__((annotate("jit"))) {
               printInt(C);
               printInt(C2);
               printInt(rc3);
               printInt(D);
             };
}

auto forwardLambda(int rc1, int rc2, int rc3) {
  int loc = 1;
  return declareLambdaThree(rc1, loc, rc3);
}

template <typename T> void run(T &&LB) {
  kernel<<<1, 1>>>(PROTEUS_REGISTER_LAMBDA(LB));
  gpuErrCheck(gpuDeviceSynchronize());
}

void forDeclareLambda() {
  for (int i = 0; i < 5; ++i) {
    auto lam = declareLambda(i, 2);
    run(lam);
  }
}

int main() {
  int Zero = 0;
  int One = 1;
  int Two = 2;

  auto ZeroLambda = declareLambda(Zero, One);

  auto OneLambda = declareLambda(One, Two);

  auto TwoLambda = declareLambda(Two, Zero);

  auto BigLam = declareLambdaThree(Zero, One, Two);

  auto ForLam = forwardLambda(Zero, One, Two);

  run(ZeroLambda);
  run(OneLambda);
  run(TwoLambda);
  run(BigLam);
  run(ForLam);

  return 0;
}

// clang-format off
// CHECK-FIRST: [LambdaSpec] Replacing slot 2 with i32 1
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 0
// CHECK: Integer = 0
// CHECK: Integer = 1
// CHECK: Integer = 5
// CHECK-FIRST: [LambdaSpec] Replacing slot 2 with i32 2
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 1
// CHECK: Integer = 1
// CHECK: Integer = 2
// CHECK: Integer = 5
// CHECK-FIRST: [LambdaSpec] Replacing slot 2 with i32 0
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 2
// CHECK: Integer = 2
// CHECK: Integer = 0
// CHECK: Integer = 5
// CHECK-FIRST: [LambdaSpec] Replacing slot 3 with i32 2
// CHECK-FIRST: [LambdaSpec] Replacing slot 2 with i32 1
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 0
// CHECK: Integer = 0
// CHECK: Integer = 1
// CHECK: Integer = 2
// CHECK: Integer = 4
// CHECK: Integer = 0
// CHECK: Integer = 1
// CHECK: Integer = 2
// CHECK: Integer = 4
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 2 NumHits 1
// CHECK-COUNT-3: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: [proteus][JitEngineDevice] StorageCache rank 0 hits 0 accesses 4
// CHECK-SECOND: [proteus][JitEngineDevice] StorageCache rank 0 hits 4 accesses 4
