// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_pointer_captures | %FILECHECK %s --implicit-check-not="[LambdaSpec] Replacing slot 0 "
// RUN: rm -rf "%t.$$.proteus"
// Pointer captures are not auto specialized, scalar captures next to them are.
// clang-format on

#include <cstdio>

#include <proteus/JitInterface.h>

int main() {
  int X = 5;
  int *P = &X;
  double D = 1.5;
  proteus::register_lambda([P, D]() __attribute__((annotate("jit"))) {
    *P += 1;
    printf("X %d D %.1f\n", *P, D);
  })();
  printf("X after %d\n", X);

  return 0;
}

// clang-format off
// CHECK: [LambdaSpec] Replacing slot 1 with double 1.500000e+00
// CHECK: X 6 D 1.5
// CHECK: X after 6
// clang-format on
