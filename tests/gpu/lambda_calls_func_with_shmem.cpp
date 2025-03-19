// RUN: rm -rf .proteus
// RUN: ./lambda_calls_func_with_shmem.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./lambda_calls_func_with_shmem.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf .proteus

#include <iostream>

#include "proteus/JitInterface.hpp"

#include "gpu_common.h"
#include <proteus/JitInterface.hpp>

template <typename T> __global__ __attribute__((annotate("jit"))) void kernel(T LB)
{
   LB();
}

template <typename T> void launch(T &&LB)
{
   proteus::register_lambda(LB);
   kernel<<<1, 1>>>(LB);
}

__device__ void square(double * v)
{
   v[0] = v[0] * v[0];
}

int main(int argc, char **argv) {
  proteus::init();

  double *x;
  gpuErrCheck(gpuMallocManaged(&x, sizeof(double) * 2));
  double a = 3.14;

  launch([=] __device__ __attribute__((annotate("jit"))) () {
    __shared__ double sm[1];
    square(sm);
  });

  std::cout << "x[0:1] = " << x[0] << "," << x[1] << "\n";

  gpuErrCheck(gpuFree(x));
}


// CHECK: x[0:1] = 9.8596,9.8596
// CHECK: JitCache hits 0 total 1
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
