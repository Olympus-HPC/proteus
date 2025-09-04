// clang-format off
// RUN: rm -rf .proteus
// RUN: PROTEUS_TRACE_OUTPUT=2 ./inlining_regression.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// PROTEUS_TRACE_OUTPUT=2 ./inlining_regression.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf .proteus
// clang-format on

#include <cstddef>
#include <cstdlib>
#include <iostream>

#include "raja_style_launch.hpp"
#include <proteus/JitInterface.hpp>

int main() {
  proteus::init();

  size_t N = 1024;
  double *X;
  double *Y;

  gpuErrCheck(gpuMallocManaged(&X, sizeof(double) * N));
  gpuErrCheck(gpuMallocManaged(&Y, sizeof(double) * N));

  for (std::size_t I{0}; I < N; I++) {
    X[I] = 0.31414 * I;
    Y[I] = 0.0;
  }

  std::cout << Y[10] << std::endl;
  double A = 6.2;
  forall(N, [=] __host__ __device__(int I) { Y[I] += X[I] * A; });
  gpuErrCheck(gpuDeviceSynchronize());
  std::cout << Y[10] << std::endl;

  forall(N, [=] __host__ __device__(int I) { Y[I] += 2 * X[I] * A; });
  gpuErrCheck(gpuDeviceSynchronize());
  std::cout << Y[10] << std::endl;

  forall(N, [=] __host__ __device__(int I) { Y[I] += 3 * X[I] * A; });
  gpuErrCheck(gpuDeviceSynchronize());
  std::cout << Y[10] << std::endl;

  forall(N, [=] __host__ __device__(int I) { Y[I] -= 6 * X[I] * A; });
  gpuErrCheck(gpuDeviceSynchronize());
  std::cout << Y[10] << std::endl;

  gpuErrCheck(gpuFree(X));
  gpuErrCheck(gpuFree(Y));

  proteus::finalize();
}

// clang-format off
// CHECK: 0
// CHECK-NOT: thread_privatize
// CHECK: 19.4767
// CHECK-NOT: thread_privatize
// CHECK: 58.43
// CHECK-NOT: thread_privatize
// CHECK: 116.86
// CHECK-NOT: thread_privatize
// CHECK: 5.17836e-15
// CHECK-NOT: thread_privatize
// CHECK: JitCache hits 0 total 4
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 4
// CHECK-SECOND: JitStorageCache hits 4 total 4
