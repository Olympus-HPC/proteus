// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_mixed_captures | %FILECHECK %s
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_AUTO_READONLY_CAPTURES=0 PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_mixed_captures | %FILECHECK %s --check-prefix=DISABLED
// RUN: rm -rf "%t.$$.proteus"
// Explicit jit_variable and auto read-only captures combine, and
// PROTEUS_AUTO_READONLY_CAPTURES=0 leaves only the explicit ones.
// clang-format on

#include <cstdio>

#include <proteus/JitInterface.h>

int main() {
  int A = 1;
  int B = 2;
  double C = 3.5;
  auto Lambda = [A = proteus::jit_variable(A), B, C]()
                    __attribute__((annotate("jit"))) {
                      printf("A %d B %d C %.1f\n", A, B, C);
                    };
  proteus::register_lambda(Lambda)();

  return 0;
}

// clang-format off
// CHECK-DAG: [LambdaSpec] Replacing slot 0 with i32 1
// CHECK-DAG: [LambdaSpec] Replacing slot 1 with i32 2
// CHECK-DAG: [LambdaSpec] Replacing slot 2 with double 3.500000e+00
// CHECK-NOT: [LambdaSpec]
// CHECK: A 1 B 2 C 3.5

// DISABLED: [LambdaSpec] Replacing slot 0 with i32 1
// DISABLED-NOT: [LambdaSpec]
// DISABLED: A 1 B 2 C 3.5
// clang-format on
