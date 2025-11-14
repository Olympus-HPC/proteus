// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT=1 %build/lambda | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/lambda | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <iostream>

#include <proteus/JitInterface.hpp>

template <typename F> void run(F &&Func) {
  proteus::register_lambda(Func);
  Func();
}

int main() {
  proteus::init();

  size_t N = 1024;
  double A = 3.14;
  double B = 1.484;

  double *X = static_cast<double *>(malloc(sizeof(double) * N));
  double *Y = static_cast<double *>(malloc(sizeof(double) * N));

  for (std::size_t I{0}; I < N; I++) {
    X[I] = 0.31414 * I;
    Y[I] = 0.0;
  }

  std::cout << Y[1] << std::endl;

  run([
    =, N = proteus::jit_variable(N), A = proteus::jit_variable(A),
    B = proteus::jit_variable(B)
  ]() __attribute__((annotate("jit"))) {
    for (std::size_t I{0}; I < N; ++I) {
      Y[I] += A * B * X[I];
    }
  });

  std::cout << Y[1] << std::endl;

  free(X);
  free(Y);

  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK: 0
// CHECK-FIRST: [LambdaSpec] Replacing slot 2 with double 1.484000e+00
// CHECK-FIRST: [LambdaSpec] Replacing slot 1 with double 3.140000e+00
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i64 1024
// CHECK: 1.46382
// CHECK: [proteus][JitEngineHost] MemoryCache rank 0 hits 0 accesses 1
// CHECK: [proteus][JitEngineHost] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: [proteus][JitEngineHost] StorageCache rank 0 hits 0 accesses 1
// CHECK-SECOND: [proteus][JitEngineHost] StorageCache rank 0 hits 1 accesses 1
