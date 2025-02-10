// RUN: rm -rf .proteus
// RUN: ./lambda | FileCheck %s --check-prefixes=CHECK
// RUN: rm -rf .proteus

#include <iostream>

#include "proteus/JitInterface.hpp"

template <typename F> void run(F &&f) {
  proteus::register_lambda(f);
  f();
}

int main(int argc, char **argv) {
  int N = 1024;
  double a = 3.14;
  double b = 1.484;

  double *x = static_cast<double *>(malloc(sizeof(double) * N));
  double *y = static_cast<double *>(malloc(sizeof(double) * N));

  for (std::size_t i{0}; i < N; i++) {
    x[i] = 0.31414 * i;
    y[i] = 0.0;
  }

  std::cout << y[1] << std::endl;

  run([
    =, N = proteus::jit_variable(N), a = proteus::jit_variable(a),
    b = proteus::jit_variable(b)
  ]() __attribute__((annotate("jit"))) {
    for (std::size_t i{0}; i < N; ++i) {
      y[i] += a * b * x[i];
    }
  });

  std::cout << y[1] << std::endl;

  free(x);
  free(y);
}

// CHECK: 0
// CHECK: 1.46382
// CHECK: JitCache hits 0 total 1
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
