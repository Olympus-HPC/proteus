// RUN: ./lambda | FileCheck %s --check-prefixes=CHECK

#include <iostream>
#include <chrono>

#include "JitVariable.hpp"


template<typename F>
void run(F&& f) {
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

  run([=, N=proteus::jit_variable(N), a=proteus::jit_variable(a), b=proteus::jit_variable(b)] ()  __attribute__((annotate("jit"))) {
    for (std::size_t i{0}; i < N; ++i) {
      y[i] += a * b * x[i];
    }
  });
  

  free(x);
  free(y);
}
