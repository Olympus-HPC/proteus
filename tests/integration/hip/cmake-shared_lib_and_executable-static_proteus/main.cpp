// RUN: ./daxpy.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./daxpy.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-SECOND
#include <cstddef>
#include <cstdlib>
#include <iostream>

#include "../../../gpu/gpu_common.h"

void daxpy(double a, double *x, double *y, int N);

int main(int argc, char **argv) {
  int N = 1024;
  double *x;
  double *y;

  gpuErrCheck(gpuMallocManaged(&x, sizeof(double) * N));
  gpuErrCheck(gpuMallocManaged(&y, sizeof(double) * N));

  for (std::size_t i{0}; i < N; i++) {
    x[i] = 0.31414 * i;
    y[i] = 0.0;
  }

  std::cout << y[10] << std::endl;
  daxpy(6.2, x, y, N);
  gpuErrCheck(gpuDeviceSynchronize());
  std::cout << y[10] << std::endl;
  daxpy(6.2, x, y, N);
  gpuErrCheck(gpuDeviceSynchronize());
  std::cout << y[10] << std::endl;

  gpuErrCheck(gpuFree(x));
  gpuErrCheck(gpuFree(y));
}

// CHECK: 0
// CHECK: 19944.1
// CHECK: 39888.2
// CHECK: JitCache hits 1 total 2
// CHECK: HashValue {{[0-9]+}} NumExecs 2 NumHits 1
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
