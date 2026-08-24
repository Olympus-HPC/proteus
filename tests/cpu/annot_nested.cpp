// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization;kernel-trace" %build/annot_nested | %FILECHECK %s
// RUN: rm -rf "%t.$$.proteus"
// Same as lambda_nested, for the annotated function interface: an annotated
// function called from a JIT'd function specializes on its own argument.
// clang-format on

#include <cstdio>

#include <proteus/JitInterface.h>

int Result;

__attribute__((annotate("jit", 1))) void innerFn(int W) { Result += W; }

__attribute__((annotate("jit", 1))) void outerFn(int V, int W) {
  Result = V * 100;
  innerFn(W);
}

int main() {
  for (int W : {10, 20, 30}) {
    outerFn(1, W);
    printf("Result %d\n", Result);
  }

  return 0;
}

// clang-format off
// CHECK: [ArgSpec] Replaced Function _Z7outerFnii ArgNo 0 with value i32 1
// CHECK: [ArgSpec] Replaced Function _Z7innerFni ArgNo 0 with value i32 10
// CHECK: Result 110
// CHECK: [ArgSpec] Replaced Function _Z7innerFni ArgNo 0 with value i32 20
// CHECK: Result 120
// CHECK: [ArgSpec] Replaced Function _Z7innerFni ArgNo 0 with value i32 30
// CHECK: Result 130
// CHECK: === Kernel Trace (rank 0) ===
// CHECK-DAG: outerFn(int, int)  rank=0  specializations=1  launches=3
// CHECK-DAG: innerFn(int)  rank=0  specializations=3  launches=3
// CHECK: === End Kernel Trace ===
// clang-format on
