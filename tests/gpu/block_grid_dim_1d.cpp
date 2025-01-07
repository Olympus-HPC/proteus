// clang-format off
// RUN: ./block_grid_dim_1d.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-FIRST 
// Second run uses the object cache. 
// RUN: ./block_grid_dim_1d.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-SECOND
// clang-format on

#include <climits>
#include <cstdio>

#include "gpu_common.h"

__global__ __attribute__((annotate("jit"))) void kernel() {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx == gridDim.x * blockDim.x - 1) {
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
    dim3 blockDim(tid * 32, 1, 1);
    dim3 gridDim(tid, 1, 1);
    kernel<<<gridDim, blockDim>>>();
  }
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// clang-format off
// CHECK:ThreadId: (31 0 0) BlockID: (0 0 0) BlockDim: (32 1 1) GridDim: (1 1 1)
// CHECK:ThreadId: (63 0 0) BlockID: (1 0 0) BlockDim: (64 1 1) GridDim: (2 1 1)
// CHECK:ThreadId: (95 0 0) BlockID: (2 0 0) BlockDim: (96 1 1) GridDim: (3 1 1)
// CHECK:ThreadId: (127 0 0) BlockID: (3 0 0) BlockDim: (128 1 1) GridDim: (4 1 1)
// CHECK:ThreadId: (159 0 0) BlockID: (4 0 0) BlockDim: (160 1 1) GridDim: (5 1 1)
// CHECK:ThreadId: (191 0 0) BlockID: (5 0 0) BlockDim: (192 1 1) GridDim: (6 1 1)
// CHECK:ThreadId: (223 0 0) BlockID: (6 0 0) BlockDim: (224 1 1) GridDim: (7 1 1)
// CHECK:ThreadId: (255 0 0) BlockID: (7 0 0) BlockDim: (256 1 1) GridDim: (8 1 1)
// CHECK:ThreadId: (287 0 0) BlockID: (8 0 0) BlockDim: (288 1 1) GridDim: (9 1 1)
// CHECK:ThreadId: (319 0 0) BlockID: (9 0 0) BlockDim: (320 1 1) GridDim: (10 1 1)
// CHECK:ThreadId: (351 0 0) BlockID: (10 0 0) BlockDim: (352 1 1) GridDim: (11 1 1)
// CHECK:ThreadId: (383 0 0) BlockID: (11 0 0) BlockDim: (384 1 1) GridDim: (12 1 1)
// CHECK:ThreadId: (415 0 0) BlockID: (12 0 0) BlockDim: (416 1 1) GridDim: (13 1 1)
// CHECK:ThreadId: (447 0 0) BlockID: (13 0 0) BlockDim: (448 1 1) GridDim: (14 1 1)
// CHECK:ThreadId: (479 0 0) BlockID: (14 0 0) BlockDim: (480 1 1) GridDim: (15 1 1)
// CHECK:ThreadId: (511 0 0) BlockID: (15 0 0) BlockDim: (512 1 1) GridDim: (16 1 1)
// CHECK:ThreadId: (543 0 0) BlockID: (16 0 0) BlockDim: (544 1 1) GridDim: (17 1 1)
// CHECK:ThreadId: (575 0 0) BlockID: (17 0 0) BlockDim: (576 1 1) GridDim: (18 1 1)
// CHECK:ThreadId: (607 0 0) BlockID: (18 0 0) BlockDim: (608 1 1) GridDim: (19 1 1)
// CHECK:ThreadId: (639 0 0) BlockID: (19 0 0) BlockDim: (640 1 1) GridDim: (20 1 1)
// CHECK:ThreadId: (671 0 0) BlockID: (20 0 0) BlockDim: (672 1 1) GridDim: (21 1 1)
// CHECK:ThreadId: (703 0 0) BlockID: (21 0 0) BlockDim: (704 1 1) GridDim: (22 1 1)
// CHECK:ThreadId: (735 0 0) BlockID: (22 0 0) BlockDim: (736 1 1) GridDim: (23 1 1)
// CHECK:ThreadId: (767 0 0) BlockID: (23 0 0) BlockDim: (768 1 1) GridDim: (24 1 1)
// CHECK:ThreadId: (799 0 0) BlockID: (24 0 0) BlockDim: (800 1 1) GridDim: (25 1 1)
// CHECK:ThreadId: (831 0 0) BlockID: (25 0 0) BlockDim: (832 1 1) GridDim: (26 1 1)
// CHECK:ThreadId: (863 0 0) BlockID: (26 0 0) BlockDim: (864 1 1) GridDim: (27 1 1)
// CHECK:ThreadId: (895 0 0) BlockID: (27 0 0) BlockDim: (896 1 1) GridDim: (28 1 1)
// CHECK:ThreadId: (927 0 0) BlockID: (28 0 0) BlockDim: (928 1 1) GridDim: (29 1 1)
// CHECK:ThreadId: (959 0 0) BlockID: (29 0 0) BlockDim: (960 1 1) GridDim: (30 1 1)
// CHECK:ThreadId: (991 0 0) BlockID: (30 0 0) BlockDim: (992 1 1) GridDim: (31 1 1)
// CHECK:ThreadId: (1023 0 0) BlockID: (31 0 0) BlockDim: (1024 1 1) GridDim: (32 1 1)

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
