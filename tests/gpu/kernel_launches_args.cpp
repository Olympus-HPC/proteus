// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/kernel_launches_args.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/kernel_launches_args.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on
#include <climits>
#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.hpp>

__global__ __attribute__((annotate("jit"))) void kernel(int A, int B) {
  A += 1;
  B += 2;
  printf("Kernel %d %d\n", A, B);
}

int main() {
  proteus::init();

  int A = 23;
  int B = 42;
  kernel<<<1, 1>>>(A, B);
  void *Args[] = {&A, &B};
  gpuErrCheck(gpuLaunchKernel((const void *)kernel, 1, 1, Args, 0, 0));
  gpuErrCheck(gpuDeviceSynchronize());

  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK: Kernel 24 44
// CHECK: [proteus][JitEngineDevice] MemoryCache procuid 0 hits 1 accesses 2
// CHECK: [proteus][JitEngineDevice] MemoryCache procuid 0 HashValue {{[0-9]+}} NumExecs 2 NumHits 1
// CHECK-FIRST: [proteus][JitEngineDevice] StorageCache procuid 0 hits 0 accesses 1
// CHECK-SECOND: [proteus][JitEngineDevice] StorageCache procuid 0 hits 1 accesses 1
