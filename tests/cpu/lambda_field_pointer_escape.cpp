// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_field_pointer_escape | %FILECHECK %s --implicit-check-not="[LambdaSpec] Replacing slot 1 "
// RUN: rm -rf "%t.$$.proteus"
// A capture whose address escapes the lambda is not specialized.
// clang-format on

#include <cstdio>

#include <proteus/JitInterface.h>

__attribute__((noinline)) void observe(const int *Ptr) {
  printf("observe %d\n", *Ptr);
}

int main() {
  int A = 1;
  int B = 2;
  int C = 3;
  proteus::register_lambda([A, B, C]() __attribute__((annotate("jit"))) {
    observe(&B);
    printf("A %d B %d C %d\n", A, B, C);
  })();

  return 0;
}

// clang-format off
// CHECK-DAG: [LambdaSpec] Replacing slot 0 with i32 1
// CHECK-DAG: [LambdaSpec] Replacing slot 2 with i32 3
// CHECK: observe 2
// CHECK: A 1 B 2 C 3
// clang-format on
