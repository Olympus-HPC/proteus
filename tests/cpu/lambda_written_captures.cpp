// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_written_captures | %FILECHECK %s --implicit-check-not="[LambdaSpec] Replacing slot 1 "
// RUN: rm -rf "%t.$$.proteus"
// A capture the lambda writes is not specialized. LambdaFunctorWrapper calls
// the lambda through a const operator(), so the write goes through const_cast
// instead of a mutable lambda.
// clang-format on

#include <cstdio>

#include <proteus/JitInterface.h>

int main() {
  int A = 10;
  int B = 20;
  int C = 30;
  auto Wrapper =
      proteus::register_lambda([A, B, C]() __attribute__((annotate("jit"))) {
        const_cast<int &>(B) += 1;
        printf("A %d B %d C %d\n", A, B, C);
      });
  Wrapper();
  Wrapper();

  return 0;
}

// clang-format off
// CHECK-DAG: [LambdaSpec] Replacing slot 0 with i32 10
// CHECK-DAG: [LambdaSpec] Replacing slot 2 with i32 30
// CHECK: A 10 B 21 C 30
// CHECK: A 10 B 22 C 30
// clang-format on
