// RUN: ./daxpy.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./daxpy.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-SECOND
#include <cstddef>
#include <cstdlib>
#include <iostream>

#include "../../../gpu/gpu_common.h"

void daxpy(double A, double *X, double *Y, int N);

int main(int argc, char **argv) {
  int N = 1024;
  double *X;
  double *Y;

  gpuErrCheck(gpuMallocManaged(&X, sizeof(double) * N));
  gpuErrCheck(gpuMallocManaged(&Y, sizeof(double) * N));

  for (std::size_t I{0}; I < N; I++) {
    X[I] = 0.31414 * I;
    Y[I] = 0.0;
  }

  std::cout << Y[10] << std::endl;
  daxpy(6.2, X, Y, N);
  gpuErrCheck(gpuDeviceSynchronize());
  std::cout << Y[10] << std::endl;
  daxpy(6.2, X, Y, N);
  gpuErrCheck(gpuDeviceSynchronize());
  std::cout << Y[10] << std::endl;

  gpuErrCheck(gpuFree(X));
  gpuErrCheck(gpuFree(Y));
}

// CHECK: 0
// CHECK: 19944.1
// CHECK: 39888.2
// CHECK: JitCache hits 1 total 2
// CHECK: HashValue {{[0-9]+}} NumExecs 2 NumHits 1
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
