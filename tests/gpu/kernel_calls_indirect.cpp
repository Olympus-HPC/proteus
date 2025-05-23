// clang-format off
// RUN: rm -rf .proteus
// RUN: ./kernel_calls_indirect.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./kernel_calls_indirect.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf .proteus
// clang-format on

#include "gpu_common.h"
#include <cstdio>
#include <proteus/JitInterface.hpp>

__device__ int inc(int Item) { return Item + 1; }

template <typename Func> __device__ int foo(Func F, int Item) {
  return F(Item);
}

__global__ __attribute__((annotate("jit"))) void kernel(int Item) {
  int Ret = foo(inc, Item);
  printf("Kernel %d\n", Ret);
}

int main() {
  proteus::init();

  kernel<<<1, 1>>>(42);
  gpuErrCheck(gpuDeviceSynchronize());

  proteus::finalize();
  return 0;
}

// CHECK: Kernel 43
// CHECK: JitCache hits 0 total 1
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
