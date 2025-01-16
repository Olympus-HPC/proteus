// clang-format off
// RUN: rm -rf .proteus
// RUN: ./block_grid_dim_2d.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-FIRST 
// Second run uses the object cache.
// RUN: ./block_grid_dim_2d.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf .proteus
// clang-format on

#include <climits>
#include <cstdio>

#include "gpu_common.h"

__global__ __attribute__((annotate("jit"))) void kernel() {
  int idx = threadIdx.y + blockIdx.y * blockDim.y;
  if (idx == gridDim.y * blockDim.y - 1) {
    printf("ThreadId: (%d %d %d) BlockID: (%d %d %d) BlockDim: (%d %d %d) "
           "GridDim: (%d %d %d)\n",
           (int)threadIdx.x, (int)threadIdx.y, (int)threadIdx.z,
           (int)blockIdx.x, (int)blockIdx.y, (int)blockIdx.z, (int)blockDim.x,
           (int)blockDim.y, (int)blockDim.z, (int)gridDim.x, (int)gridDim.y,
           (int)gridDim.z);
  }
}

int main() {
  for (int tid = 1; tid <= 32; tid++) {
    dim3 blockDim(1, tid * 32, 1);
    dim3 gridDim(1, tid, 1);
    kernel<<<gridDim, blockDim>>>();
  }
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// clang-format off
// CHECK:ThreadId: (0 31 0) BlockID: (0 0 0) BlockDim: (1 32 1) GridDim: (1 1 1)
// CHECK:ThreadId: (0 63 0) BlockID: (0 1 0) BlockDim: (1 64 1) GridDim: (1 2 1)
// CHECK:ThreadId: (0 95 0) BlockID: (0 2 0) BlockDim: (1 96 1) GridDim: (1 3 1)
// CHECK:ThreadId: (0 127 0) BlockID: (0 3 0) BlockDim: (1 128 1) GridDim: (1 4 1)
// CHECK:ThreadId: (0 159 0) BlockID: (0 4 0) BlockDim: (1 160 1) GridDim: (1 5 1)
// CHECK:ThreadId: (0 191 0) BlockID: (0 5 0) BlockDim: (1 192 1) GridDim: (1 6 1)
// CHECK:ThreadId: (0 223 0) BlockID: (0 6 0) BlockDim: (1 224 1) GridDim: (1 7 1)
// CHECK:ThreadId: (0 255 0) BlockID: (0 7 0) BlockDim: (1 256 1) GridDim: (1 8 1)
// CHECK:ThreadId: (0 287 0) BlockID: (0 8 0) BlockDim: (1 288 1) GridDim: (1 9 1)
// CHECK:ThreadId: (0 319 0) BlockID: (0 9 0) BlockDim: (1 320 1) GridDim: (1 10 1)
// CHECK:ThreadId: (0 351 0) BlockID: (0 10 0) BlockDim: (1 352 1) GridDim: (1 11 1)
// CHECK:ThreadId: (0 383 0) BlockID: (0 11 0) BlockDim: (1 384 1) GridDim: (1 12 1)
// CHECK:ThreadId: (0 415 0) BlockID: (0 12 0) BlockDim: (1 416 1) GridDim: (1 13 1)
// CHECK:ThreadId: (0 447 0) BlockID: (0 13 0) BlockDim: (1 448 1) GridDim: (1 14 1)
// CHECK:ThreadId: (0 479 0) BlockID: (0 14 0) BlockDim: (1 480 1) GridDim: (1 15 1)
// CHECK:ThreadId: (0 511 0) BlockID: (0 15 0) BlockDim: (1 512 1) GridDim: (1 16 1)
// CHECK:ThreadId: (0 543 0) BlockID: (0 16 0) BlockDim: (1 544 1) GridDim: (1 17 1)
// CHECK:ThreadId: (0 575 0) BlockID: (0 17 0) BlockDim: (1 576 1) GridDim: (1 18 1)
// CHECK:ThreadId: (0 607 0) BlockID: (0 18 0) BlockDim: (1 608 1) GridDim: (1 19 1)
// CHECK:ThreadId: (0 639 0) BlockID: (0 19 0) BlockDim: (1 640 1) GridDim: (1 20 1)
// CHECK:ThreadId: (0 671 0) BlockID: (0 20 0) BlockDim: (1 672 1) GridDim: (1 21 1)
// CHECK:ThreadId: (0 703 0) BlockID: (0 21 0) BlockDim: (1 704 1) GridDim: (1 22 1)
// CHECK:ThreadId: (0 735 0) BlockID: (0 22 0) BlockDim: (1 736 1) GridDim: (1 23 1)
// CHECK:ThreadId: (0 767 0) BlockID: (0 23 0) BlockDim: (1 768 1) GridDim: (1 24 1)
// CHECK:ThreadId: (0 799 0) BlockID: (0 24 0) BlockDim: (1 800 1) GridDim: (1 25 1)
// CHECK:ThreadId: (0 831 0) BlockID: (0 25 0) BlockDim: (1 832 1) GridDim: (1 26 1)
// CHECK:ThreadId: (0 863 0) BlockID: (0 26 0) BlockDim: (1 864 1) GridDim: (1 27 1)
// CHECK:ThreadId: (0 895 0) BlockID: (0 27 0) BlockDim: (1 896 1) GridDim: (1 28 1)
// CHECK:ThreadId: (0 927 0) BlockID: (0 28 0) BlockDim: (1 928 1) GridDim: (1 29 1)
// CHECK:ThreadId: (0 959 0) BlockID: (0 29 0) BlockDim: (1 960 1) GridDim: (1 30 1)
// CHECK:ThreadId: (0 991 0) BlockID: (0 30 0) BlockDim: (1 992 1) GridDim: (1 31 1)
// CHECK:ThreadId: (0 1023 0) BlockID: (0 31 0) BlockDim: (1 1024 1) GridDim: (1 32 1)

//CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
//CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
//CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
//CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
//CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
//CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
//CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
//CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
//CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
//CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
//CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
//CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
//CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
//CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
//CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
//CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
//CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
//CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
//CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
//CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
//CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
//CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
//CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
//CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
//CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
//CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
//CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
//CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
//CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
//CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
//CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
//CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 32
// CHECK-SECOND: JitStorageCache hits 32 total 32
// clang-format on
