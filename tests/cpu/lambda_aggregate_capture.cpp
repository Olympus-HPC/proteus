// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_aggregate_capture | %FILECHECK %s --implicit-check-not="with i32 2" --implicit-check-not="with double 3.5"
// RUN: rm -rf "%t.$$.proteus"
// An aggregate captured by value is not auto specialized, the scalar captures
// around it are. A lambda returning an aggregate through sret still computes
// the right result.
// clang-format on

#include <cstdio>

#include <proteus/JitInterface.h>

struct Pair {
  int X;
  double Y;
};

struct Triple {
  long A;
  long B;
  long C;
};

int main() {
  int A = 1;
  Pair S{2, 3.5};
  int B = 4;
  proteus::register_lambda([A, S, B]() __attribute__((annotate("jit"))) {
    printf("A %d S.X %d S.Y %.1f B %d\n", A, S.X, S.Y, B);
  })();

  int X = 5;
  long Y = 9;
  Triple R =
      proteus::register_lambda([X, Y]() __attribute__((annotate("jit"))) {
        return Triple{X, Y, X + Y};
      })();
  printf("R %ld %ld %ld\n", R.A, R.B, R.C);

  return 0;
}

// clang-format off
// CHECK-DAG: [LambdaSpec] Replacing slot 0 with i32 1
// CHECK-DAG: [LambdaSpec] Replacing slot {{[0-9]+}} with i32 4
// CHECK: A 1 S.X 2 S.Y 3.5 B 4
// CHECK: R 5 9 14
// clang-format on
