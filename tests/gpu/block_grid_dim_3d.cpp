// RUN: ./block_grid_dim_3d.%ext | FileCheck %s
// --check-prefixes=CHECK,CHECK-FIRST Second run uses the object cache. RUN:
// ./block_grid_dim_3d.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-SECOND

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
  for (int tid = 1; tid <= 32; tid++) {
    dim3 blockDim(1, 1, tid * 32);
    dim3 gridDim(1, 1, tid);
    kernel<<<gridDim, blockDim>>>();
  }
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// clang-format off
// CHECK:ThreadId: (0 0 31) BlockID: (0 0 0) BlockDim: (1 1 32) GridDim: (1 1 1)
// CHECK:ThreadId: (0 0 63) BlockID: (0 0 1) BlockDim: (1 1 64) GridDim: (1 1 2)
// CHECK:ThreadId: (0 0 95) BlockID: (0 0 2) BlockDim: (1 1 96) GridDim: (1 1 3)
// CHECK:ThreadId: (0 0 127) BlockID: (0 0 3) BlockDim: (1 1 128) GridDim: (1 1 4)
// CHECK:ThreadId: (0 0 159) BlockID: (0 0 4) BlockDim: (1 1 160) GridDim: (1 1 5)
// CHECK:ThreadId: (0 0 191) BlockID: (0 0 5) BlockDim: (1 1 192) GridDim: (1 1 6)
// CHECK:ThreadId: (0 0 223) BlockID: (0 0 6) BlockDim: (1 1 224) GridDim: (1 1 7)
// CHECK:ThreadId: (0 0 255) BlockID: (0 0 7) BlockDim: (1 1 256) GridDim: (1 1 8)
// CHECK:ThreadId: (0 0 287) BlockID: (0 0 8) BlockDim: (1 1 288) GridDim: (1 1 9)
// CHECK:ThreadId: (0 0 319) BlockID: (0 0 9) BlockDim: (1 1 320) GridDim: (1 1 10)
// CHECK:ThreadId: (0 0 351) BlockID: (0 0 10) BlockDim: (1 1 352) GridDim: (1 1 11)
// CHECK:ThreadId: (0 0 383) BlockID: (0 0 11) BlockDim: (1 1 384) GridDim: (1 1 12)
// CHECK:ThreadId: (0 0 415) BlockID: (0 0 12) BlockDim: (1 1 416) GridDim: (1 1 13)
// CHECK:ThreadId: (0 0 447) BlockID: (0 0 13) BlockDim: (1 1 448) GridDim: (1 1 14)
// CHECK:ThreadId: (0 0 479) BlockID: (0 0 14) BlockDim: (1 1 480) GridDim: (1 1 15)
// CHECK:ThreadId: (0 0 511) BlockID: (0 0 15) BlockDim: (1 1 512) GridDim: (1 1 16)
// CHECK:ThreadId: (0 0 543) BlockID: (0 0 16) BlockDim: (1 1 544) GridDim: (1 1 17)
// CHECK:ThreadId: (0 0 575) BlockID: (0 0 17) BlockDim: (1 1 576) GridDim: (1 1 18)
// CHECK:ThreadId: (0 0 607) BlockID: (0 0 18) BlockDim: (1 1 608) GridDim: (1 1 19)
// CHECK:ThreadId: (0 0 639) BlockID: (0 0 19) BlockDim: (1 1 640) GridDim: (1 1 20)
// CHECK:ThreadId: (0 0 671) BlockID: (0 0 20) BlockDim: (1 1 672) GridDim: (1 1 21)
// CHECK:ThreadId: (0 0 703) BlockID: (0 0 21) BlockDim: (1 1 704) GridDim: (1 1 22)
// CHECK:ThreadId: (0 0 735) BlockID: (0 0 22) BlockDim: (1 1 736) GridDim: (1 1 23)
// CHECK:ThreadId: (0 0 767) BlockID: (0 0 23) BlockDim: (1 1 768) GridDim: (1 1 24)
// CHECK:ThreadId: (0 0 799) BlockID: (0 0 24) BlockDim: (1 1 800) GridDim: (1 1 25)
// CHECK:ThreadId: (0 0 831) BlockID: (0 0 25) BlockDim: (1 1 832) GridDim: (1 1 26)
// CHECK:ThreadId: (0 0 863) BlockID: (0 0 26) BlockDim: (1 1 864) GridDim: (1 1 27)
// CHECK:ThreadId: (0 0 895) BlockID: (0 0 27) BlockDim: (1 1 896) GridDim: (1 1 28)
// CHECK:ThreadId: (0 0 927) BlockID: (0 0 28) BlockDim: (1 1 928) GridDim: (1 1 29)
// CHECK:ThreadId: (0 0 959) BlockID: (0 0 29) BlockDim: (1 1 960) GridDim: (1 1 30)
// CHECK:ThreadId: (0 0 991) BlockID: (0 0 30) BlockDim: (1 1 992) GridDim: (1 1 31)
// CHECK:ThreadId: (0 0 1023) BlockID: (0 0 31) BlockDim: (1 1 1024) GridDim: (1 1 32)

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
