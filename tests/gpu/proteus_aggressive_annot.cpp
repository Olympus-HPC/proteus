#include <climits>
#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.hpp>

__global__ void kernel_one() { printf("Kernel_one\n"); }

__device__ int inc(int Item) { return Item + 1; }

template <typename Func> __device__ int foo(Func F, int Item) {
  return F(Item);
}

__global__ void kernel_two(int Item) {
  int Ret = foo(inc, Item);
  printf("Kernel two %d\n", Ret);
}

__global__ void kernel_three() { printf("Kernel three\n"); }

template <typename T> gpuError_t launcher(T KernelIn) {
  return gpuLaunchKernel((const void *)KernelIn, 1, 1, 0, 0, 0);
}

int main() {
  proteus::init();

  kernel_one<<<1, 1>>>();
  gpuErrCheck(gpuDeviceSynchronize());

  kernel_two<<<1, 1>>>(1);
  gpuErrCheck(gpuDeviceSynchronize());

  launcher(kernel_three);
  gpuErrCheck(gpuDeviceSynchronize());

  proteus::finalize();
  return 0;
}

// CHECK: Kernel gvar 24 addr [[ADDR:[a-z0-9]+]]
// CHECK: Kernel2 gvar 25 addr [[ADDR]]
// CHECK: Kernel3 gvar 26 addr [[ADDR]]
// CHECK: JitCache hits 0 total 2
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 2
// CHECK-SECOND: JitStorageCache hits 2 total 2
