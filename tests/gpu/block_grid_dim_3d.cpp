// clang-format off
// RUN: rm -rf .proteus
// RUN: PROTEUS_TRACE_OUTPUT=1 ./block_grid_dim_3d.%ext |  %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN:./block_grid_dim_3d.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf .proteus
// clang-format on

#include <climits>
#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.hpp>

__global__ __attribute__((annotate("jit"))) void kernel() {
  unsigned int Idx = threadIdx.z + blockIdx.z * blockDim.z;
  if (Idx == (gridDim.z * blockDim.z - 1)) {
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

  // Maximum BlockSize in the z-dimension is 64 for CUDA, hence the reduced
  // iteration space.
  for (int Tid = 4; Tid <= 64; Tid *= 2) {
    dim3 BlockDim(1, 1, Tid);
    dim3 GridDim(1, 1, Tid / 4);
    kernel<<<GridDim, BlockDim>>>();
    gpuErrCheck(gpuDeviceSynchronize());
  }

  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK-FIRST: [DimSpec] Replace call to {{_ZL20__hip_get_grid_dim_xv|llvm.nvvm.read.ptx.sreg.nctaid.x}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZL20__hip_get_grid_dim_yv|llvm.nvvm.read.ptx.sreg.nctaid.y}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZL20__hip_get_grid_dim_zv|llvm.nvvm.read.ptx.sreg.nctaid.z}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZL21__hip_get_block_dim_xv|llvm.nvvm.read.ptx.sreg.ntid.x}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZL21__hip_get_block_dim_yv|llvm.nvvm.read.ptx.sreg.ntid.y}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZL21__hip_get_block_dim_zv|llvm.nvvm.read.ptx.sreg.ntid.z}} with constant i32 4
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [LaunchBoundSpec] GridSize 1 BlockSize 4
// CHECK: ThreadId: (0 0 3) BlockID: (0 0 0) BlockDim: (1 1 4) GridDim: (1 1 1)
// CHECK-FIRST: [DimSpec] Replace call to {{_ZL20__hip_get_grid_dim_xv|llvm.nvvm.read.ptx.sreg.nctaid.x}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZL20__hip_get_grid_dim_yv|llvm.nvvm.read.ptx.sreg.nctaid.y}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZL20__hip_get_grid_dim_zv|llvm.nvvm.read.ptx.sreg.nctaid.z}} with constant i32 2
// CHECK-FIRST: [DimSpec] Replace call to {{_ZL21__hip_get_block_dim_xv|llvm.nvvm.read.ptx.sreg.ntid.x}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZL21__hip_get_block_dim_yv|llvm.nvvm.read.ptx.sreg.ntid.y}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZL21__hip_get_block_dim_zv|llvm.nvvm.read.ptx.sreg.ntid.z}} with constant i32 8
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [LaunchBoundSpec] GridSize 2 BlockSize 8
// CHECK: ThreadId: (0 0 7) BlockID: (0 0 1) BlockDim: (1 1 8) GridDim: (1 1 2)
// CHECK-FIRST: [DimSpec] Replace call to {{_ZL20__hip_get_grid_dim_xv|llvm.nvvm.read.ptx.sreg.nctaid.x}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZL20__hip_get_grid_dim_yv|llvm.nvvm.read.ptx.sreg.nctaid.y}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZL20__hip_get_grid_dim_zv|llvm.nvvm.read.ptx.sreg.nctaid.z}} with constant i32 4
// CHECK-FIRST: [DimSpec] Replace call to {{_ZL21__hip_get_block_dim_xv|llvm.nvvm.read.ptx.sreg.ntid.x}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZL21__hip_get_block_dim_yv|llvm.nvvm.read.ptx.sreg.ntid.y}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZL21__hip_get_block_dim_zv|llvm.nvvm.read.ptx.sreg.ntid.z}} with constant i32 16
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [LaunchBoundSpec] GridSize 4 BlockSize 16
// CHECK: ThreadId: (0 0 15) BlockID: (0 0 3) BlockDim: (1 1 16) GridDim: (1 1 4)
// CHECK-FIRST: [DimSpec] Replace call to {{_ZL20__hip_get_grid_dim_xv|llvm.nvvm.read.ptx.sreg.nctaid.x}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZL20__hip_get_grid_dim_yv|llvm.nvvm.read.ptx.sreg.nctaid.y}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZL20__hip_get_grid_dim_zv|llvm.nvvm.read.ptx.sreg.nctaid.z}} with constant i32 8
// CHECK-FIRST: [DimSpec] Replace call to {{_ZL21__hip_get_block_dim_xv|llvm.nvvm.read.ptx.sreg.ntid.x}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZL21__hip_get_block_dim_yv|llvm.nvvm.read.ptx.sreg.ntid.y}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZL21__hip_get_block_dim_zv|llvm.nvvm.read.ptx.sreg.ntid.z}} with constant i32 32
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [LaunchBoundSpec] GridSize 8 BlockSize 32
// CHECK: ThreadId: (0 0 31) BlockID: (0 0 7) BlockDim: (1 1 32) GridDim: (1 1 8)
// CHECK-FIRST: [DimSpec] Replace call to {{_ZL20__hip_get_grid_dim_xv|llvm.nvvm.read.ptx.sreg.nctaid.x}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZL20__hip_get_grid_dim_yv|llvm.nvvm.read.ptx.sreg.nctaid.y}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZL20__hip_get_grid_dim_zv|llvm.nvvm.read.ptx.sreg.nctaid.z}} with constant i32 16
// CHECK-FIRST: [DimSpec] Replace call to {{_ZL21__hip_get_block_dim_xv|llvm.nvvm.read.ptx.sreg.ntid.x}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZL21__hip_get_block_dim_yv|llvm.nvvm.read.ptx.sreg.ntid.y}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZL21__hip_get_block_dim_zv|llvm.nvvm.read.ptx.sreg.ntid.z}} with constant i32 64
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [LaunchBoundSpec] GridSize 16 BlockSize 64
// CHECK: ThreadId: (0 0 63) BlockID: (0 0 15) BlockDim: (1 1 64) GridDim: (1 1 16)
// CHECK-COUNT-5: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 5
// CHECK-SECOND: JitStorageCache hits 5 total 5
