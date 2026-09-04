// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization;kernel-trace" %build/lambda_nested | %FILECHECK %s
// RUN: rm -rf "%t.$$.proteus"
// A lambda registered inside a JIT'd lambda body specializes on its own
// runtime constants: the outer region is compiled once for V, and the inner one
// is recompiled for each W.
// clang-format on

#include <cstdio>

#include <proteus/JitInterface.h>

template <typename F> void run(F &&Func) { proteus::register_lambda(Func)(); }

void nested(int V, int W) {
  run([=, V = proteus::jit_variable(V)]() __attribute__((annotate("jit"))) {
    run([=, W = proteus::jit_variable(W)]()
            __attribute__((annotate("jit"))) { printf("V %d W %d\n", V, W); });
  });
}

int main() {
  nested(1, 10);
  nested(1, 20);
  nested(1, 30);

  return 0;
}

// clang-format off
// CHECK: [LambdaSpec] Replacing slot 0 with i32 1
// CHECK: [LambdaSpec] Replacing slot 0 with i32 10
// CHECK: V 1 W 10
// CHECK: [LambdaSpec] Replacing slot 0 with i32 20
// CHECK: V 1 W 20
// CHECK: [LambdaSpec] Replacing slot 0 with i32 30
// CHECK: V 1 W 30
// CHECK: === Kernel Trace (rank 0) ===
// CHECK-DAG: nested(int, int)::$_0::operator()() const  rank=0  specializations=1  launches=3
// CHECK-DAG: nested(int, int)::$_0::operator()() const::{{.*}}operator()() const  rank=0  specializations=3  launches=3
// CHECK: === End Kernel Trace ===
// clang-format on
