// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT=1 %build/indirect_launcher_arg.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/indirect_launcher_arg.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <climits>
#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.hpp>

__global__ __attribute__((annotate("jit", 1))) void kernel(int A) {
  printf("Kernel %d\n", A);
}

template <typename T> gpuError_t launcher(T KernelIn, int A) {
  void *Args[] = {&A};
  return gpuLaunchKernel((const void *)KernelIn, 1, 1, Args, 0, 0);
}

int main() {
  proteus::init();

  kernel<<<1, 1>>>(42);
  gpuErrCheck(gpuDeviceSynchronize());
  gpuErrCheck(launcher(kernel, 24));
  gpuErrCheck(gpuDeviceSynchronize());

  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK-FIRST: [ArgSpec] Replaced Function _Z6kerneli ArgNo 0 with value i32 42
// CHECK-FIRST: [LaunchBoundSpec] MaxThreads 1 MinBlocksPerSM 0
// CHECK: Kernel 42
// CHECK-FIRST: [ArgSpec] Replaced Function _Z6kerneli ArgNo 0 with value i32 24
// CHECK-FIRST: [LaunchBoundSpec] MaxThreads 1 MinBlocksPerSM 0
// CHECK: Kernel 24
// CHECK: JitCache hits 0 total 2
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 2
// CHECK-SECOND: JitStorageCache hits 2 total 2
