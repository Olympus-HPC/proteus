// RUN: rm -rf .proteus
// RUN: ./shared_array.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./shared_array.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf .proteus
#include <climits>
#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.hpp>

template <typename Lambda>
__global__ __attribute__((annotate("jit"))) void kernel(Lambda &&Body) {
  printf("Kernel\n");
  Body();
}

template <typename Lambda> void launcher(Lambda &&Body) {
  kernel<<<1, 1>>>(Body);
  gpuErrCheck(gpuDeviceSynchronize());
}

int main() {
  proteus::init();

  int Dims = 3;
  launcher(proteus::register_lambda(
      [=, Dims = proteus::jit_variable(Dims)] __device__() {
        double *Array = proteus::shared_array<double>(Dims);
        Array[0] = 1.0;
        Array[1] = 2.0;
        Array[2] = 3.0;
        printf("Lambda Array[0] %lf Array[1] %lf Array[2] %lf\n", Array[0],
               Array[1], Array[2]);
      }));

  proteus::finalize();
  return 0;
}

// CHECK: Kernel
// CHECK: Lambda Array[0] 1.000000 Array[1] 2.000000 Array[2] 3.000000
// CHECK: JitCache hits 0 total 1
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
