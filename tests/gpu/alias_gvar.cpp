// clang-format off
// RUN: rm -rf .proteus
// RUN: ./alias_gvar.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./alias_gvar.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf .proteus
// clang-format on

#include "gpu_common.h"
#include <iostream>

// Global device variable
__device__ int original_data[1024];

// Create an alias using extern declaration with attribute
extern "C" __device__ int alias_data[1024]
    __attribute__((alias("original_data")));

// Kernel that uses both the original and alias
__global__ __attribute__((annotate("jit"))) void kernel(int *output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < 1024) {
    // Write to original
    original_data[idx] = idx * 2;

    // Read from alias (should be the same memory)
    output[idx] = alias_data[idx];
  }
}

// Host function to demonstrate usage
int main() {
  int *output;

  // Allocate managed device memory
  gpuErrCheck(gpuMallocManaged(&output, 1024 * sizeof(int)));

  // Launch kernel
  dim3 block(256);
  dim3 grid((1024 + block.x - 1) / block.x);

  kernel<<<grid, block>>>(output);
  gpuErrCheck(gpuDeviceSynchronize());

  // Verify first few elements
  std::cout << "First 5 elements: ";
  for (int i = 0; i < 5; i++) {
    std::cout << output[i] << " ";
  }
  std::cout << std::endl;

  // Cleanup
  gpuErrCheck(gpuFree(output));

  return 0;
}

// CHECK: First 5 elements: 0 2 4 6 8
// CHECK: JitCache hits 0 total 1
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
