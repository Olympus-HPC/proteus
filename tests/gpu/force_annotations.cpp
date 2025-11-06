// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/force_annotations.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/force_annotations.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <climits>
#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.hpp>

__global__ void kernel1() { printf("Kernel one\n"); }

__device__ int inc(int Item) { return Item + 1; }

template <typename Func> __device__ int foo(Func F, int Item) {
  return F(Item);
}

__global__ void kernel2(int Item) {
  int Ret = foo(inc, Item);
  printf("Kernel two %d\n", Ret);
}

__global__ void kernel3() { printf("Kernel three\n"); }

template <typename T> gpuError_t launcher(T KernelIn) {
  return gpuLaunchKernel((const void *)KernelIn, 1, 1, 0, 0, 0);
}

int main() {
  proteus::init();

  kernel1<<<1, 1>>>();
  gpuErrCheck(gpuDeviceSynchronize());

  kernel2<<<1, 1>>>(1);
  gpuErrCheck(gpuDeviceSynchronize());

  gpuErrCheck(launcher(kernel3));
  gpuErrCheck(gpuDeviceSynchronize());

  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK: Kernel one
// CHECK: Kernel two 2
// CHECK: Kernel three
// CHECK: JitCache hits 0 total 3
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 3
// CHECK-SECOND: JitStorageCache hits 3 total 3
