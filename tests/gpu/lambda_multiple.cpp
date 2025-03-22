// clang-format off
// RUN: rm -rf .proteus
// RUN: ./lambda_multiple.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./lambda_multiple.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf .proteus
// clang-format on

#include <iostream>

#include "gpu_common.h"
#include <proteus/JitInterface.hpp>

template <typename T>
__global__ __attribute__((annotate("jit"))) void kernel(T LB) {
  LB();
}

__global__ __attribute__((annotate("jit"))) void kernelSimple() {
  printf("Kernel simple\n");
}

template <typename T> void run(T &&LB) {
  kernel<<<1, 1>>>(LB);
  gpuErrCheck(gpuDeviceSynchronize());
}

void lambdaCaller(int V) {
  run(proteus::register_lambda(
      [=, V = proteus::jit_variable(V)] __device__() { printf("V %d\n", V); }));
}

int main(int argc, char **argv) {
  proteus::init();
  // We expect that lambdas will specialize and NOT hit the cache since its
  // kernel invocation is templated on the unique lambda type.  The
  // non-templated kernelSimple should hit the cache as it is independent of the
  // lambda (type and JIT variables).
  lambdaCaller(1);
  kernelSimple<<<1, 1>>>();
  gpuErrCheck(gpuDeviceSynchronize());
  lambdaCaller(2);

  lambdaCaller(1);
  kernelSimple<<<1, 1>>>();
  gpuErrCheck(gpuDeviceSynchronize());
  lambdaCaller(2);

  proteus::finalize();
}

// CHECK: V 1
// CHECK: Kernel simple
// CHECK: V 2
// CHECK: V 1
// CHECK: Kernel simple
// CHECK: V 2
// CHECK: JitCache hits 3 total 6
// CHECK-3: HashValue {{[0-9]+}} NumExecs 2 NumHits 1
// CHECK-FIRST: JitStorageCache hits 0 total 3
// CHECK-SECOND: JitStorageCache hits 3 total 3
