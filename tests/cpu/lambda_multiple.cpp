// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT=1 %build/lambda_multiple | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/lambda_multiple | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <iostream>

#include "proteus/JitInterface.h"

#include <proteus/JitInterface.h>

template <typename F> void run(F &&Func) {
  proteus::register_lambda(Func);
  Func();
}

void lambdaCaller(int V) {
  run([ =, V = proteus::jit_variable(V) ]()
          __attribute__((annotate("jit"))) { printf("V %d\n", V); });
}

__attribute__((annotate("jit", 1))) void foo(int A) { printf("foo %d\n", A); }

int main() {
  proteus::init();
  // We expect that lambdas will specialize and NOT hit the cache since its
  // kernel invocation is templated on the unique lambda type.  The
  // non-templated kernelSimple should hit the cache as it is independent of the
  // lambda (type and JIT variables).
  lambdaCaller(1);
  foo(42);
  lambdaCaller(2);

  lambdaCaller(1);
  foo(42);
  lambdaCaller(2);

  proteus::finalize();
}

// clang-format off
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 1
// CHECK: V 1
// CHECK-FIRST: [ArgSpec] Replaced Function _Z3fooi ArgNo 0 with value i32 42
// CHECK: foo 42
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 2
// CHECK: V 2
// CHECK: V 1
// CHECK: foo 42
// CHECK: V 2
// CHECK: [proteus][JitEngineHost] MemoryCache rank 0 hits 3 accesses 6
// CHECK-COUNT-3: [proteus][JitEngineHost] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 2 NumHits 1
// CHECK-FIRST: [proteus][JitEngineHost] StorageCache rank 0 hits 0 accesses 3
// CHECK-SECOND: [proteus][JitEngineHost] StorageCache rank 0 hits 3 accesses 3
