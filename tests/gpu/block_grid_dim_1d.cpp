// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_SPECIALIZE_DIMS_ASSUME=1  PROTEUS_TRACE_OUTPUT=1 %build/block_grid_dim_1d.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/block_grid_dim_1d.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <climits>
#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.hpp>

__global__ __attribute__((annotate("jit"))) void kernel() {
  unsigned int Idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (Idx == (gridDim.x * blockDim.x - 1)) {
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

  for (int Tid = 64; Tid <= 1024; Tid *= 2) {
    dim3 BlockDim(Tid, 1, 1);
    dim3 GridDim(Tid / 64, 1, 1);
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
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI14__HIP_BlockDimE3__XcvjEv|_ZL21__hip_get_block_dim_xv|llvm.nvvm.read.ptx.sreg.ntid.x}} with constant i32 64
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI14__HIP_BlockDimE3__YcvjEv|_ZL21__hip_get_block_dim_yv|llvm.nvvm.read.ptx.sreg.ntid.y}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI14__HIP_BlockDimE3__ZcvjEv|_ZL21__hip_get_block_dim_zv|llvm.nvvm.read.ptx.sreg.ntid.z}} with constant i32 1
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [LaunchBoundSpec] MaxThreads 64 MinBlocksPerSM 0
// CHECK: ThreadId: (63 0 0) BlockID: (0 0 0) BlockDim: (64 1 1) GridDim: (1 1 1)
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI13__HIP_GridDimE3__XcvjEv|_ZL20__hip_get_grid_dim_xv|llvm.nvvm.read.ptx.sreg.nctaid.x}} with constant i32 2
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI13__HIP_GridDimE3__YcvjEv|_ZL20__hip_get_grid_dim_yv|llvm.nvvm.read.ptx.sreg.nctaid.y}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI13__HIP_GridDimE3__ZcvjEv|_ZL20__hip_get_grid_dim_zv|llvm.nvvm.read.ptx.sreg.nctaid.z}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI14__HIP_BlockDimE3__XcvjEv|_ZL21__hip_get_block_dim_xv|llvm.nvvm.read.ptx.sreg.ntid.x}} with constant i32 128
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI14__HIP_BlockDimE3__YcvjEv|_ZL21__hip_get_block_dim_yv|llvm.nvvm.read.ptx.sreg.ntid.y}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI14__HIP_BlockDimE3__ZcvjEv|_ZL21__hip_get_block_dim_zv|llvm.nvvm.read.ptx.sreg.ntid.z}} with constant i32 1
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [LaunchBoundSpec] MaxThreads 128 MinBlocksPerSM 0
// CHECK: ThreadId: (127 0 0) BlockID: (1 0 0) BlockDim: (128 1 1) GridDim: (2 1 1)
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI13__HIP_GridDimE3__XcvjEv|_ZL20__hip_get_grid_dim_xv|llvm.nvvm.read.ptx.sreg.nctaid.x}} with constant i32 4
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI13__HIP_GridDimE3__YcvjEv|_ZL20__hip_get_grid_dim_yv|llvm.nvvm.read.ptx.sreg.nctaid.y}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI13__HIP_GridDimE3__ZcvjEv|_ZL20__hip_get_grid_dim_zv|llvm.nvvm.read.ptx.sreg.nctaid.z}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI14__HIP_BlockDimE3__XcvjEv|_ZL21__hip_get_block_dim_xv|llvm.nvvm.read.ptx.sreg.ntid.x}} with constant i32 256
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI14__HIP_BlockDimE3__YcvjEv|_ZL21__hip_get_block_dim_yv|llvm.nvvm.read.ptx.sreg.ntid.y}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI14__HIP_BlockDimE3__ZcvjEv|_ZL21__hip_get_block_dim_zv|llvm.nvvm.read.ptx.sreg.ntid.z}} with constant i32 1
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [LaunchBoundSpec] MaxThreads 256 MinBlocksPerSM 0
// CHECK: ThreadId: (255 0 0) BlockID: (3 0 0) BlockDim: (256 1 1) GridDim: (4 1 1)
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI13__HIP_GridDimE3__XcvjEv|_ZL20__hip_get_grid_dim_xv|llvm.nvvm.read.ptx.sreg.nctaid.x}} with constant i32 8
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI13__HIP_GridDimE3__YcvjEv|_ZL20__hip_get_grid_dim_yv|llvm.nvvm.read.ptx.sreg.nctaid.y}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI13__HIP_GridDimE3__ZcvjEv|_ZL20__hip_get_grid_dim_zv|llvm.nvvm.read.ptx.sreg.nctaid.z}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI14__HIP_BlockDimE3__XcvjEv|_ZL21__hip_get_block_dim_xv|llvm.nvvm.read.ptx.sreg.ntid.x}} with constant i32 512
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI14__HIP_BlockDimE3__YcvjEv|_ZL21__hip_get_block_dim_yv|llvm.nvvm.read.ptx.sreg.ntid.y}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI14__HIP_BlockDimE3__ZcvjEv|_ZL21__hip_get_block_dim_zv|llvm.nvvm.read.ptx.sreg.ntid.z}} with constant i32 1
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [LaunchBoundSpec] MaxThreads 512 MinBlocksPerSM 0
// CHECK: ThreadId: (511 0 0) BlockID: (7 0 0) BlockDim: (512 1 1) GridDim: (8 1 1)
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI13__HIP_GridDimE3__XcvjEv|_ZL20__hip_get_grid_dim_xv|llvm.nvvm.read.ptx.sreg.nctaid.x}} with constant i32 16
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI13__HIP_GridDimE3__YcvjEv|_ZL20__hip_get_grid_dim_yv|llvm.nvvm.read.ptx.sreg.nctaid.y}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI13__HIP_GridDimE3__ZcvjEv|_ZL20__hip_get_grid_dim_zv|llvm.nvvm.read.ptx.sreg.nctaid.z}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI14__HIP_BlockDimE3__XcvjEv|_ZL21__hip_get_block_dim_xv|llvm.nvvm.read.ptx.sreg.ntid.x}} with constant i32 1024
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI14__HIP_BlockDimE3__YcvjEv|_ZL21__hip_get_block_dim_yv|llvm.nvvm.read.ptx.sreg.ntid.y}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI14__HIP_BlockDimE3__ZcvjEv|_ZL21__hip_get_block_dim_zv|llvm.nvvm.read.ptx.sreg.ntid.z}} with constant i32 1
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [DimSpec]
// CHECK-FIRST: [LaunchBoundSpec] MaxThreads 1024 MinBlocksPerSM 0
// CHECK: ThreadId: (1023 0 0) BlockID: (15 0 0) BlockDim: (1024 1 1) GridDim: (16 1 1)
// CHECK-COUNT-5: [proteus][JitEngineDevice] MemoryCache procuid 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: [proteus][JitEngineDevice] StorageCache procuid 0 hits 0 accesses 5
// CHECK-SECOND: [proteus][JitEngineDevice] StorageCache procuid 0 hits 5 accesses 5
