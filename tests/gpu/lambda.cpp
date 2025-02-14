// RUN: rm -rf .proteus
// RUN: ./lambda.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./lambda.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf .proteus

#include <iostream>

#include "proteus/JitInterface.hpp"

#include "gpu_common.h"

template <typename T>
__global__ __attribute__((annotate("jit"))) void kernel(T LB) {
  std::size_t i = blockIdx.x + threadIdx.x;
  if (i == 0)
    LB();
}

template <typename T>
__global__ __attribute__((annotate("jit"))) void kernel(int n, T LB) {
  std::size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n)
    LB(i);
}

template <typename T> void register_run(T &&LB) {
  proteus::register_lambda(LB);
  kernel<<<1, 1>>>(LB);
  gpuErrCheck(gpuDeviceSynchronize());
}

template <typename T> void register_run(int n, T &&LB) {
  proteus::register_lambda(LB);
  constexpr int block_size = 256;
  const int num_blocks = (n + block_size - 1) / block_size;
  kernel<<<num_blocks, block_size>>>(n, LB);
  gpuErrCheck(gpuDeviceSynchronize());
}

template <typename T> void run(T &&LB) {
  kernel<<<1, 1>>>(LB);
  gpuErrCheck(gpuDeviceSynchronize());
}

inline void launch(double c, double *x) {
  register_run([ =, c = proteus::jit_variable(c) ] __device__
               __attribute__((annotate("jit"))) () { x[0] = c + 2; });
  std::cout << "x[0] = " << x[0] << "\n";
}

int main(int argc, char **argv) {
  double a{3.14};
  double b{1.23};
  double c{4.56};
  double *x;
  gpuErrCheck(gpuMallocManaged(&x, sizeof(double) * 2));

  register_run([ =, a = proteus::jit_variable(a) ] __device__
               __attribute__((annotate("jit"))) () { x[0] = a; });

  std::cout << "x[0] = " << x[0] << "\n";

  register_run(
      2, [ =, c = proteus::jit_variable(c) ] __device__
      __attribute__((annotate("jit"))) (int i) { x[i] = c; });

  std::cout << "x[0:1] = " << x[0] << "," << x[1] << "\n";

  run(proteus::register_lambda([
    =, a = proteus::jit_variable(a), b = proteus::jit_variable(b)
  ] __device__ __attribute__((annotate("jit"))) () { x[0] = a + b; }));
  std::cout << "x[0] = " << x[0] << "\n";

  launch(c, x);
  gpuErrCheck(gpuFree(x));
}

// CHECK: x[0] = 3.14
// CHECK: x[0:1] = 4.56,4.56
// CHECK: x[0] = 4.37
// CHECK: x[0] = 6.56
// CHECK: JitCache hits 0 total 4
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 4
// CHECK-SECOND: JitStorageCache hits 4 total 4
