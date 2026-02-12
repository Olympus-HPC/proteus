// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/daxpy_annot_long | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/daxpy_annot_long | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstddef>
#include <cstdlib>
#include <iostream>

#include <proteus/JitInterface.h>

__attribute__((annotate("jit", 1, R"({"arg":4})"))) void
myDaxpy(double A, double *X, double *Y, size_t N) {
  for (std::size_t I{0}; I < N; I++) {
    Y[I] += X[I] * A;
  }
}

int main() {

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
// CHECK: [proteus][JitEngineHost] MemoryCache rank 0 hits 0 accesses 2
// CHECK: [proteus][JitEngineHost] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: [proteus][JitEngineHost] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: [proteus][JitEngineHost] StorageCache rank 0 hits 0 accesses 2
// CHECK-SECOND: [proteus][JitEngineHost] StorageCache rank 0 hits 2 accesses 2
