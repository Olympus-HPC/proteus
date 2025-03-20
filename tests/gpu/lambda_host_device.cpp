// clang-format off
// RUN: rm -rf .proteus
// RUN: ./lambda_host_device.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./lambda_host_device.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf .proteus
// clang-format on

#include <iostream>

#include "gpu_common.h"
#include <proteus/JitInterface.hpp>

template <typename T>
__global__ __attribute__((annotate("jit"))) void kernel(T LB) {
  LB();
}

template <typename T> void launcher(T &&LB) {
  proteus::register_lambda(LB);
  kernel<<<1, 1>>>(LB);
  gpuErrCheck(gpuDeviceSynchronize());
  LB();
}

int main(int argc, char **argv) {
  proteus::init();
  int A = 42;
  launcher([ =, A = proteus::jit_variable(A) ] __host__ __device__()
               __attribute__((annotate("jit"))) { printf("Lambda %d\n", A); });
  proteus::finalize();
}

// CHECK: Lambda 42
// CHECK: Lambda 42
// CHECK-2: JitCache hits 0 total 1
// CHECK-2: HashValue {{[0-9]+}} NumExecs 2 NumHits 1
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
