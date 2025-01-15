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

__global__ __attribute__((annotate("jit"))) void kernel() {
  int idx = threadIdx.z + blockIdx.z * blockDim.z;
  if (idx == gridDim.z * blockDim.z - 1) {
    printf("ThreadId: (%d %d %d) BlockID: (%d %d %d) BlockDim: (%d %d %d) "
           "GridDim: (%d %d %d)\n",
           (int)threadIdx.x, (int)threadIdx.y, (int)threadIdx.z,
           (int)blockIdx.x, (int)blockIdx.y, (int)blockIdx.z, (int)blockDim.x,
           (int)blockDim.y, (int)blockDim.z, (int)gridDim.x, (int)gridDim.y,
           (int)gridDim.z);
  }
}

int main() {
  for (int tid = 1; tid <= 2; tid++) {
    dim3 blockDim(1, 1, tid * 32);
    dim3 gridDim(1, 1, tid);
    kernel<<<gridDim, blockDim>>>();
    gpuErrCheck(gpuDeviceSynchronize());
  }
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
