// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization;cache-stats" %build/lambda_def | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="cache-stats" %build/lambda_def | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>

#include <proteus/JitInterface.h>

template <typename F> void registerRun(F &&Func) {
  PROTEUS_REGISTER_LAMBDA(Func)();
}

template <typename F> void run(F &&Func) { Func(); }

int main() {
  int A = 42;
  auto Lambda =
      [=, A = proteus::jit_variable(A)]()
          __attribute__((annotate("jit"))) { printf("Lambda A %d\n", A); };
  auto RegisteredLambda = PROTEUS_REGISTER_LAMBDA(Lambda);
  run(RegisteredLambda);
  run(RegisteredLambda);
  run(RegisteredLambda);

  registerRun(Lambda);
  registerRun(Lambda);
  registerRun(Lambda);

  return 0;
}

// clang-format off
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 42
// CHECK-COUNT-3: Lambda A 42
// CHECK: [proteus][JitEngineHost] MemoryCache rank 0 hits 2 accesses 3
// CHECK: [proteus][JitEngineHost] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 3 NumHits 2
// CHECK-FIRST: [proteus][JitEngineHost] StorageCache rank 0 hits 0 accesses 1
// CHECK-SECOND: [proteus][JitEngineHost] StorageCache rank 0 hits 1 accesses 1
