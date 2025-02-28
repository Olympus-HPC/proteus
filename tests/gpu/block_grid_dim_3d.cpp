// clang-format off
// RUN: rm -rf .proteus
// RUN: ./block_grid_dim_3d.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN:./block_grid_dim_3d.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf .proteus
// clang-format on

#include <climits>
#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.hpp>

__global__ __attribute__((annotate("jit"))) void kernel() {
  int Idx = threadIdx.z + blockIdx.z * blockDim.z;
  if (Idx == gridDim.z * blockDim.z - 1) {
    printf("ThreadId: (%d %d %d) BlockID: (%d %d %d) BlockDim: (%d %d %d) "
           "GridDim: (%d %d %d)\n",
           (int)threadIdx.x, (int)threadIdx.y, (int)threadIdx.z,
           (int)blockIdx.x, (int)blockIdx.y, (int)blockIdx.z, (int)blockDim.x,
           (int)blockDim.y, (int)blockDim.z, (int)gridDim.x, (int)gridDim.y,
           (int)gridDim.z);
  }
}

int main() {
  proteus::init();

  for (int Tid = 1; Tid <= 2; Tid++) {
    dim3 BlockDim(1, 1, Tid * 32);
    dim3 GridDim(1, 1, Tid);
    kernel<<<GridDim, BlockDim>>>();
    gpuErrCheck(gpuDeviceSynchronize());
  }

  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK:ThreadId: (0 0 31) BlockID: (0 0 0) BlockDim: (1 1 32) GridDim: (1 1 1)
// CHECK:ThreadId: (0 0 63) BlockID: (0 0 1) BlockDim: (1 1 64) GridDim: (1 1 2)

//CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
//CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 2
// CHECK-SECOND: JitStorageCache hits 2 total 2
// clang-format on
