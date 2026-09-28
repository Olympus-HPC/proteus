// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization;kernel-trace" %build/lambda_auto_relaunch | %FILECHECK %s
// RUN: rm -rf "%t.$$.proteus"
// Each distinct auto captured value gets its own specialization, and a
// repeated value reuses it.
// clang-format on

#include <cstdio>

#include <proteus/JitInterface.h>

int main() {
  int Scale = 10;
  for (int I = 0; I < 6; ++I) {
    int V = I % 3;
    proteus::register_lambda([V, Scale]() __attribute__((annotate("jit"))) {
      printf("V %d Scaled %d\n", V, V * Scale);
    })();
  }

  return 0;
}

// clang-format off
// CHECK: [LambdaSpec] Replacing slot 0 with i32 0
// CHECK: V 0 Scaled 0
// CHECK: [LambdaSpec] Replacing slot 0 with i32 1
// CHECK: V 1 Scaled 10
// CHECK: [LambdaSpec] Replacing slot 0 with i32 2
// CHECK: V 2 Scaled 20
// CHECK-NOT: [LambdaSpec]
// CHECK: V 0 Scaled 0
// CHECK-NOT: [LambdaSpec]
// CHECK: V 1 Scaled 10
// CHECK-NOT: [LambdaSpec]
// CHECK: V 2 Scaled 20
// CHECK: === Kernel Trace (rank 0) ===
// CHECK: main::$_0::operator()() const  rank=0  specializations=3  launches=6
// CHECK: === End Kernel Trace ===
// clang-format on
