// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT=1 %build/daxpy_api | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/daxpy_api | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstddef>
#include <cstdlib>
#include <iostream>

#include <proteus/JitInterface.hpp>

void myDaxpy(double A, double *X, double *Y, size_t N) {
  proteus::jit_arg(A);
  proteus::jit_arg(N);
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

// clang-format off
// CHECK: 0
// CHECK-FIRST: [ArgSpec] Replaced Function _Z7myDaxpydPdS_m ArgNo 0 with value double 6.200000e+00
// CHECK-FIRST: [ArgSpec] Replaced Function _Z7myDaxpydPdS_m ArgNo 3 with value i64 1024
// CHECK: 19.4767
// CHECK-FIRST: [ArgSpec] Replaced Function _Z7myDaxpydPdS_m ArgNo 0 with value double 1.000000e+00
// CHECK-FIRST: [ArgSpec] Replaced Function _Z7myDaxpydPdS_m ArgNo 3 with value i64 1024
// CHECK: 22.6181
// CHECK: JitCache hits 0 total 2
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 2
// CHECK-SECOND: JitStorageCache hits 2 total 2
