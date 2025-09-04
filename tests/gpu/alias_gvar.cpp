// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT=1 %build/alias_gvar.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/alias_gvar.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include "gpu_common.h"
#include <iostream>

// Global device variable
__device__ int OriginalData[1024];

// Create an alias using extern declaration with attribute
extern "C" __device__ int AliasData[1024]
    __attribute__((alias("OriginalData")));

// Kernel that uses both the original and alias
__global__ __attribute__((annotate("jit"))) void kernel(int *Output) {
  int Idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (Idx < 1024) {
    // Write to original
    OriginalData[Idx] = Idx * 2;

    // Read from alias (should be the same memory)
    Output[Idx] = AliasData[Idx];
  }
}

// Host function to demonstrate usage
int main() {
  int *Output;

  // Allocate managed device memory
  gpuErrCheck(gpuMallocManaged(&Output, 1024 * sizeof(int)));

  // Launch kernel
  dim3 Block(256);
  dim3 Grid((1024 + Block.x - 1) / Block.x);

  kernel<<<Grid, Block>>>(Output);
  gpuErrCheck(gpuDeviceSynchronize());

  // Verify first few elements
  std::cout << "First 5 elements: ";
  for (int I = 0; I < 5; I++) {
    std::cout << Output[I] << " ";
  }
  std::cout << std::endl;

  // Cleanup
  gpuErrCheck(gpuFree(Output));

  return 0;
}

// clang-format off
// CHECK: First 5 elements: 0 2 4 6 8
// CHECK: JitCache hits 0 total 1
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
