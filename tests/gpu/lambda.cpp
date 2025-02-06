// RUN: rm -rf .proteus
// RUN: ./lambda.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./lambda.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf .proteus

#include <iostream>

#include "proteus/JitVariable.hpp"

#include "gpu_common.h"

template <typename T>
__global__ __attribute__((annotate("jit"))) void kernel(T LB) {
  std::size_t i = blockIdx.x + threadIdx.x;
  if (i == 0)
    LB();
}

template <typename T> void run(T &&LB) {
  proteus::register_lambda(LB);
  kernel<<<1, 1>>>(LB);
  gpuErrCheck(gpuDeviceSynchronize());
}

int main(int argc, char **argv) {
  double a{3.14};
  double *x;
  gpuErrCheck(gpuMallocManaged(&x, sizeof(double) * 2));

  run([ =, a = proteus::jit_variable(a) ] __device__
      __attribute__((annotate("jit"))) () { x[0] = a; });

  std::cout << "x[0] = " << x[0] << "\n";
  gpuErrCheck(gpuFree(x));
}

// CHECK: x[0] = 3.14
// CHECK: JitCache hits 0 total 1
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
