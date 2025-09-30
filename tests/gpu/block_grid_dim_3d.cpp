// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_SPECIALIZE_DIMS_ASSUME=1 PROTEUS_TRACE_OUTPUT=1 %build/block_grid_dim_3d.%ext |  %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/block_grid_dim_3d.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
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
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI13__HIP_GridDimE3__XcvjEv|_ZL20__hip_get_grid_dim_xv|llvm.nvvm.read.ptx.sreg.nctaid.x}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI13__HIP_GridDimE3__YcvjEv|_ZL20__hip_get_grid_dim_yv|llvm.nvvm.read.ptx.sreg.nctaid.y}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI13__HIP_GridDimE3__ZcvjEv|_ZL20__hip_get_grid_dim_zv|llvm.nvvm.read.ptx.sreg.nctaid.z}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI14__HIP_BlockDimE3__XcvjEv|_ZL21__hip_get_block_dim_xv|llvm.nvvm.read.ptx.sreg.ntid.x}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI14__HIP_BlockDimE3__YcvjEv|_ZL21__hip_get_block_dim_yv|llvm.nvvm.read.ptx.sreg.ntid.y}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI14__HIP_BlockDimE3__ZcvjEv|_ZL21__hip_get_block_dim_zv|llvm.nvvm.read.ptx.sreg.ntid.z}} with constant i32 4
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [LaunchBoundSpec] MaxThreads 4 BlocksPerEU 0
// CHECK: ThreadId: (0 0 3) BlockID: (0 0 0) BlockDim: (1 1 4) GridDim: (1 1 1)
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI13__HIP_GridDimE3__XcvjEv|_ZL20__hip_get_grid_dim_xv|llvm.nvvm.read.ptx.sreg.nctaid.x}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI13__HIP_GridDimE3__YcvjEv|_ZL20__hip_get_grid_dim_yv|llvm.nvvm.read.ptx.sreg.nctaid.y}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI13__HIP_GridDimE3__ZcvjEv|_ZL20__hip_get_grid_dim_zv|llvm.nvvm.read.ptx.sreg.nctaid.z}} with constant i32 2
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI14__HIP_BlockDimE3__XcvjEv|_ZL21__hip_get_block_dim_xv|llvm.nvvm.read.ptx.sreg.ntid.x}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI14__HIP_BlockDimE3__YcvjEv|_ZL21__hip_get_block_dim_yv|llvm.nvvm.read.ptx.sreg.ntid.y}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI14__HIP_BlockDimE3__ZcvjEv|_ZL21__hip_get_block_dim_zv|llvm.nvvm.read.ptx.sreg.ntid.z}} with constant i32 8
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [LaunchBoundSpec] MaxThreads 8 BlocksPerEU 0
// CHECK: ThreadId: (0 0 7) BlockID: (0 0 1) BlockDim: (1 1 8) GridDim: (1 1 2)
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI13__HIP_GridDimE3__XcvjEv|_ZL20__hip_get_grid_dim_xv|llvm.nvvm.read.ptx.sreg.nctaid.x}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI13__HIP_GridDimE3__YcvjEv|_ZL20__hip_get_grid_dim_yv|llvm.nvvm.read.ptx.sreg.nctaid.y}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI13__HIP_GridDimE3__ZcvjEv|_ZL20__hip_get_grid_dim_zv|llvm.nvvm.read.ptx.sreg.nctaid.z}} with constant i32 4
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI14__HIP_BlockDimE3__XcvjEv|_ZL21__hip_get_block_dim_xv|llvm.nvvm.read.ptx.sreg.ntid.x}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI14__HIP_BlockDimE3__YcvjEv|_ZL21__hip_get_block_dim_yv|llvm.nvvm.read.ptx.sreg.ntid.y}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI14__HIP_BlockDimE3__ZcvjEv|_ZL21__hip_get_block_dim_zv|llvm.nvvm.read.ptx.sreg.ntid.z}} with constant i32 16
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [LaunchBoundSpec] MaxThreads 16 BlocksPerEU 0
// CHECK: ThreadId: (0 0 15) BlockID: (0 0 3) BlockDim: (1 1 16) GridDim: (1 1 4)
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI13__HIP_GridDimE3__XcvjEv|_ZL20__hip_get_grid_dim_xv|llvm.nvvm.read.ptx.sreg.nctaid.x}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI13__HIP_GridDimE3__YcvjEv|_ZL20__hip_get_grid_dim_yv|llvm.nvvm.read.ptx.sreg.nctaid.y}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI13__HIP_GridDimE3__ZcvjEv|_ZL20__hip_get_grid_dim_zv|llvm.nvvm.read.ptx.sreg.nctaid.z}} with constant i32 8
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI14__HIP_BlockDimE3__XcvjEv|_ZL21__hip_get_block_dim_xv|llvm.nvvm.read.ptx.sreg.ntid.x}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI14__HIP_BlockDimE3__YcvjEv|_ZL21__hip_get_block_dim_yv|llvm.nvvm.read.ptx.sreg.ntid.y}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI14__HIP_BlockDimE3__ZcvjEv|_ZL21__hip_get_block_dim_zv|llvm.nvvm.read.ptx.sreg.ntid.z}} with constant i32 32
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [LaunchBoundSpec] MaxThreads 32 BlocksPerEU 0
// CHECK: ThreadId: (0 0 31) BlockID: (0 0 7) BlockDim: (1 1 32) GridDim: (1 1 8)
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI13__HIP_GridDimE3__XcvjEv|_ZL20__hip_get_grid_dim_xv|llvm.nvvm.read.ptx.sreg.nctaid.x}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI13__HIP_GridDimE3__YcvjEv|_ZL20__hip_get_grid_dim_yv|llvm.nvvm.read.ptx.sreg.nctaid.y}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI13__HIP_GridDimE3__ZcvjEv|_ZL20__hip_get_grid_dim_zv|llvm.nvvm.read.ptx.sreg.nctaid.z}} with constant i32 16
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI14__HIP_BlockDimE3__XcvjEv|_ZL21__hip_get_block_dim_xv|llvm.nvvm.read.ptx.sreg.ntid.x}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI14__HIP_BlockDimE3__YcvjEv|_ZL21__hip_get_block_dim_yv|llvm.nvvm.read.ptx.sreg.ntid.y}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI14__HIP_BlockDimE3__ZcvjEv|_ZL21__hip_get_block_dim_zv|llvm.nvvm.read.ptx.sreg.ntid.z}} with constant i32 64
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [LaunchBoundSpec] MaxThreads 64 BlocksPerEU 0
// CHECK: ThreadId: (0 0 63) BlockID: (0 0 15) BlockDim: (1 1 64) GridDim: (1 1 16)
// CHECK-COUNT-5: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 5
// CHECK-SECOND: JitStorageCache hits 5 total 5
