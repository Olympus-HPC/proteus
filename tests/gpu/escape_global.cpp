// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/escape_global.%ext | %FILECHECK %s --check-prefixes=CHECK
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/escape_global.%ext | %FILECHECK %s --check-prefixes=CHECK
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <climits>
#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

__constant__ int Value;

__global__ __attribute__((annotate("jit"))) static void kernel(int *value) {
  printf("Kernel global value is %d\n", *value);
}

int main() {
  int HValue = 101;
  void *DAddr;
  gpuErrCheck(gpuGetSymbolAddress(&DAddr, Value));
  gpuErrCheck(gpuMemcpy(DAddr, &HValue, sizeof(int), gpuMemcpyHostToDevice));
  kernel<<<1, 1>>>((int *)DAddr);
  gpuErrCheck(gpuDeviceSynchronize());

  return 0;
}

// clang-format off
// CHECK: Kernel global value is 101
