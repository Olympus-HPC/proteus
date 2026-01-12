// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/kernels_gvar.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/kernels_gvar.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <climits>
#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

__device__ int Gvar = 23;

__global__ __attribute__((annotate("jit"))) void kernel() {
  Gvar++;
  printf("Kernel gvar %d addr %p\n", Gvar, &Gvar);
}

__global__ __attribute__((annotate("jit"))) void kernel2() {
  Gvar++;
  printf("Kernel2 gvar %d addr %p\n", Gvar, &Gvar);
}

__global__ void kernel3() {
  Gvar++;
  printf("Kernel3 gvar %d addr %p\n", Gvar, &Gvar);
}

int main() {
  proteus::init();

  kernel<<<1, 1>>>();
  gpuErrCheck(gpuDeviceSynchronize());
  kernel2<<<1, 1>>>();
  gpuErrCheck(gpuDeviceSynchronize());
  kernel3<<<1, 1>>>();
  gpuErrCheck(gpuDeviceSynchronize());

  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK: Kernel gvar 24 addr [[ADDR:[a-z0-9]+]]
// CHECK: Kernel2 gvar 25 addr [[ADDR]]
// CHECK: Kernel3 gvar 26 addr [[ADDR]]
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 hits 0 accesses 2
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: [proteus][JitEngineDevice] StorageCache rank 0 hits 0 accesses 2
// CHECK-SECOND: [proteus][JitEngineDevice] StorageCache rank 0 hits 2 accesses 2
