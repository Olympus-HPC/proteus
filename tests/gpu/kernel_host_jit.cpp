// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/kernel_host_jit.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/kernel_host_jit.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <climits>
#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.hpp>

__global__ void kernel() { printf("Kernel\n"); }

template <typename T>
__attribute__((annotate("jit"))) gpuError_t launcher(T KernelIn) {
  return gpuLaunchKernel((const void *)KernelIn, 1, 1, 0, 0, 0);
}

int main() {
  proteus::init();

  kernel<<<1, 1>>>();
  gpuErrCheck(launcher(kernel));
  gpuErrCheck(gpuDeviceSynchronize());

  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK: Kernel
// CHECK: Kernel
// CHECK: [proteus][JitEngineHost] MemoryCache rank 0 hits 0 accesses 1
// CHECK: [proteus][JitEngineHost] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: [proteus][JitEngineHost] StorageCache rank 0 hits 0 accesses 1
// CHECK-SECOND: [proteus][JitEngineHost] StorageCache rank 0 hits 1 accesses 1
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 hits 0 accesses 0
// CHECK: [proteus][JitEngineDevice] StorageCache rank 0 hits 0 accesses 0
