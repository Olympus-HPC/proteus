// clang-format off
// RUN: rm -rf .proteus
// RUN: PROTEUS_TRACE_OUTPUT=1 ./kernel_host_device_jit_api.%ext | %FILECHECK %s --check-prefixes=CHECK
// RUN: rm -rf .proteus
// clang-format on

#include <climits>
#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.hpp>

__global__ void kernel(int Arg) {
  proteus::jit_arg(Arg);
  printf("Kernel %d\n", Arg);
}

template <typename T>
__attribute__((annotate("jit"))) gpuError_t launcher(T KernelIn) {
  int Val = 23;
  void *Args[] = {&Val};
  return gpuLaunchKernel((const void *)KernelIn, 1, 1, Args, 0, 0);
}

int main() {
  proteus::init();

  kernel<<<1, 1>>>(42);
  gpuErrCheck(gpuDeviceSynchronize());
  gpuErrCheck(launcher(kernel));
  gpuErrCheck(gpuDeviceSynchronize());

  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK: [ArgSpec] Replaced Function _Z6kerneli ArgNo 0 with value i32 42
// CHECK: [LaunchBoundSpec] GridSize 1 BlockSize 1
// CHECK: Kernel 42
// CHECK: [ArgSpec] Replaced Function _Z6kerneli ArgNo 0 with value i32 23
// CHECK: [LaunchBoundSpec] GridSize 1 BlockSize 1
// CHECK: Kernel 23
// CHECK: JitCache hits 0 total 1
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: JitCache hits 0 total 2
// CHECK-COUNT-2: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: JitStorageCache hits 0 total 2
