// clang-format off
// RUN: rm -rf %t.$$.proteus
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT=1 %build/lambda_multiple_api | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/lambda_multiple_api | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf %t.$$.proteus
// clang-format on

#include <cstdio>

#include <proteus/JitInterface.hpp>

template <typename F> void run(F &&Func) {
  proteus::register_lambda(Func);
  Func();
}

void lambdaCaller(int V) {
  run([ =, V = proteus::jit_variable(V) ]()
          __attribute__((annotate("jit"))) { printf("V %d\n", V); });
}

void foo(int A) {
  proteus::jit_arg(A);
  printf("foo %d\n", A);
}

int main() {
  proteus::init();

  lambdaCaller(1);
  foo(42);
  lambdaCaller(2);

  lambdaCaller(1);
  foo(42);
  lambdaCaller(2);

  proteus::finalize();
}

// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 1
// CHECK: V 1
// CHECK-FIRST: [ArgSpec] Replaced Function _Z3fooi ArgNo 0 with value i32 42
// CHECK: foo 42
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 2
// CHECK: V 2
// CHECK: V 1
// CHECK: foo 42
// CHECK: V 2
// CHECK: JitCache hits 3 total 6
// CHECK-COUNT-3: HashValue {{[0-9]+}} NumExecs 2 NumHits 1
// CHECK-FIRST: JitStorageCache hits 0 total 3
// CHECK-SECOND: JitStorageCache hits 3 total 3
