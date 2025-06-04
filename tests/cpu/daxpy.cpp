// RUN: rm -rf .proteus
// RUN: ./daxpy | %FILECHECK %s --check-prefixes=CHECK
// RUN: rm -rf .proteus

#include <cstddef>
#include <cstdlib>
#include <iostream>

#include <proteus/JitInterface.hpp>

__attribute__((annotate("jit", 1, 4))) void myDaxpy(double A, double *X,
                                                    double *Y, size_t N) {
  for (std::size_t I{0}; I < N; I++) {
    Y[I] += X[I] * A;
  }
}

int main() {
  proteus::init();

  size_t N = 1024;
  double *X = static_cast<double *>(malloc(sizeof(double) * N));
  double *Y = static_cast<double *>(malloc(sizeof(double) * N));

  for (std::size_t I{0}; I < N; I++) {
    X[I] = 0.31414 * I;
    Y[I] = 0.0;
  }

  std::cout << Y[10] << std::endl;
  myDaxpy(6.2, X, Y, N);
  std::cout << Y[10] << std::endl;
  myDaxpy(1.0, X, Y, N);
  std::cout << Y[10] << std::endl;

  free(X);
  free(Y);

  proteus::finalize();
  return 0;
}

// CHECK: 0
// CHECK: 19.4767
// CHECK: 22.6181
// CHECK: JitCache hits 0 total 2
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
