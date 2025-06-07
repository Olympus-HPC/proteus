// RUN: rm -rf .proteus
// RUN: PROTEUS_TRACE_OUTPUT=1 ./lambda | %FILECHECK %s --check-prefixes=CHECK
// RUN: rm -rf .proteus

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
// CHECK: [LambdaSpec] Replacing slot 2 with double 1.484000e+00
// CHECK: [LambdaSpec] Replacing slot 1 with double 3.140000e+00
// CHECK: [LambdaSpec] Replacing slot 0 with i64 1024
// CHECK: 1.46382
// CHECK: JitCache hits 0 total 1
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
