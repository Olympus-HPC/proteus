// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_SPECIALIZE_DIMS_RANGE=0 PROTEUS_SPECIALIZE_DIMS=0 PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization;cache-stats" %build/dim_spec_cache_test.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_SPECIALIZE_DIMS_RANGE=0 PROTEUS_SPECIALIZE_DIMS=0 PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization;cache-stats" %build/dim_spec_cache_test.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_SPECIALIZE_DIMS_RANGE=1 PROTEUS_SPECIALIZE_DIMS=1 PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization;cache-stats" %build/dim_spec_cache_test.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-THIRD
// Fourth run uses the object cache.
// RUN: PROTEUS_SPECIALIZE_DIMS_RANGE=1 PROTEUS_SPECIALIZE_DIMS=1 PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization;cache-stats" %build/dim_spec_cache_test.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FOURTH
// clang-format on

#include <climits>
#include <iostream>
#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

__global__ __attribute__((annotate("jit"))) void printGridDim() {
  unsigned int Idx = threadIdx.x + threadIdx.y + threadIdx.z +
     blockIdx.x * blockDim.x + blockIdx.y * blockDim.y +
     blockIdx.z * blockDim.z;

  if (Idx == 0) {
    printf("Block dim.x = %d\n", blockDim.x);
    printf("Block dim.y = %d\n", blockDim.y);
    printf("Block dim.z = %d\n", blockDim.z);

    printf("Grid dim.x = %d\n", gridDim.x);
    printf("Grid dim.y = %d\n", gridDim.y);
    printf("Grid dim.z = %d\n", gridDim.z);
  }
}

int main() {
  dim3 GridDim4(4, 4, 4);
  dim3 BlockDim4(4, 4, 4);

  dim3 GridDim8(8, 8, 8);
  dim3 BlockDim2(2, 2, 2);

  printGridDim<<<GridDim4, BlockDim2>>>();
  gpuErrCheck(gpuDeviceSynchronize());

  printGridDim<<<GridDim8, BlockDim2>>>();
  gpuErrCheck(gpuDeviceSynchronize());

  printGridDim<<<GridDim4, BlockDim4>>>();
  gpuErrCheck(gpuDeviceSynchronize());


  return 0;
}

// clang-format off
// CHECK-THIRD: [DimSpec] Replace call to _ZL20__hip_get_grid_dim_xv with constant i32 4
// CHECK-THIRD: [DimSpec] Replace call to _ZL20__hip_get_grid_dim_yv with constant i32 4
// CHECK-THIRD: [DimSpec] Replace call to _ZL20__hip_get_grid_dim_zv with constant i32 4
// CHECK-THIRD: [DimSpec] Replace call to _ZL21__hip_get_block_dim_xv with constant i32 2
// CHECK-THIRD: [DimSpec] Replace call to _ZL21__hip_get_block_dim_yv with constant i32 2
// CHECK-THIRD: [DimSpec] Replace call to _ZL21__hip_get_block_dim_zv with constant i32 2
// CHECK-THIRD: [DimSpec] Range llvm.amdgcn.workitem.id.x [0,2)
// CHECK-THIRD: [DimSpec] Range llvm.amdgcn.workitem.id.y [0,2)
// CHECK-THIRD: [DimSpec] Range llvm.amdgcn.workitem.id.z [0,2)
// CHECK-THIRD: [DimSpec] Range llvm.amdgcn.workgroup.id.x [0,4)
// CHECK-THIRD: [DimSpec] Range llvm.amdgcn.workgroup.id.x [0,4)
// CHECK-THIRD: [DimSpec] Range llvm.amdgcn.workgroup.id.y [0,4)
// CHECK-THIRD: [DimSpec] Range llvm.amdgcn.workgroup.id.y [0,4)
// CHECK-THIRD: [DimSpec] Range llvm.amdgcn.workgroup.id.z [0,4)
// CHECK-THIRD: [DimSpec] Range llvm.amdgcn.workgroup.id.z [0,4)
// CHECK-FIRST: [LaunchBoundSpec] MaxThreads 8 MinBlocksPerSM 0
// CHECK-THIRD: [LaunchBoundSpec] MaxThreads 8 MinBlocksPerSM 0
// CHECK-SECOND: [ObjectCacheChain] Hit at level 0 (Storage) for hash {{[0-9]+}}
// CHECK-FOURTH: [ObjectCacheChain] Hit at level 0 (Storage) for hash {{[0-9]+}}
// CHECK: Block dim.x = 2
// CHECK: Block dim.y = 2
// CHECK: Block dim.z = 2
// CHECK: Grid dim.x = 4
// CHECK: Grid dim.y = 4
// CHECK: Grid dim.z = 4
// CHECK-THIRD: [DimSpec] Replace call to _ZL20__hip_get_grid_dim_xv with constant i32 8
// CHECK-THIRD: [DimSpec] Replace call to _ZL20__hip_get_grid_dim_yv with constant i32 8
// CHECK-THIRD: [DimSpec] Replace call to _ZL20__hip_get_grid_dim_zv with constant i32 8
// CHECK-THIRD: [DimSpec] Replace call to _ZL21__hip_get_block_dim_xv with constant i32 2
// CHECK-THIRD: [DimSpec] Replace call to _ZL21__hip_get_block_dim_yv with constant i32 2
// CHECK-THIRD: [DimSpec] Replace call to _ZL21__hip_get_block_dim_zv with constant i32 2
// CHECK-THIRD: [DimSpec] Range llvm.amdgcn.workitem.id.x [0,2)
// CHECK-THIRD: [DimSpec] Range llvm.amdgcn.workitem.id.y [0,2)
// CHECK-THIRD: [DimSpec] Range llvm.amdgcn.workitem.id.z [0,2)
// CHECK-THIRD: [DimSpec] Range llvm.amdgcn.workgroup.id.x [0,8)
// CHECK-THIRD: [DimSpec] Range llvm.amdgcn.workgroup.id.x [0,8)
// CHECK-THIRD: [DimSpec] Range llvm.amdgcn.workgroup.id.y [0,8)
// CHECK-THIRD: [DimSpec] Range llvm.amdgcn.workgroup.id.y [0,8)
// CHECK-THIRD: [DimSpec] Range llvm.amdgcn.workgroup.id.z [0,8)
// CHECK-THIRD: [DimSpec] Range llvm.amdgcn.workgroup.id.z [0,8)
// CHECK-THIRD: [LaunchBoundSpec] MaxThreads 8 MinBlocksPerSM 0
// CHECK-FOURTH: [ObjectCacheChain] Hit at level 0 (Storage) for hash {{[0-9]+}}
// CHECK: Block dim.x = 2
// CHECK: Block dim.y = 2
// CHECK: Block dim.z = 2
// CHECK: Grid dim.x = 8
// CHECK: Grid dim.y = 8
// CHECK: Grid dim.z = 8
// CHECK-THIRD: [DimSpec] Replace call to _ZL20__hip_get_grid_dim_xv with constant i32 4
// CHECK-THIRD: [DimSpec] Replace call to _ZL20__hip_get_grid_dim_yv with constant i32 4
// CHECK-THIRD: [DimSpec] Replace call to _ZL20__hip_get_grid_dim_zv with constant i32 4
// CHECK-THIRD: [DimSpec] Replace call to _ZL21__hip_get_block_dim_xv with constant i32 4
// CHECK-THIRD: [DimSpec] Replace call to _ZL21__hip_get_block_dim_yv with constant i32 4
// CHECK-THIRD: [DimSpec] Replace call to _ZL21__hip_get_block_dim_zv with constant i32 4
// CHECK-THIRD: [DimSpec] Range llvm.amdgcn.workitem.id.x [0,4)
// CHECK-THIRD: [DimSpec] Range llvm.amdgcn.workitem.id.y [0,4)
// CHECK-THIRD: [DimSpec] Range llvm.amdgcn.workitem.id.z [0,4)
// CHECK-THIRD: [DimSpec] Range llvm.amdgcn.workgroup.id.x [0,4)
// CHECK-THIRD: [DimSpec] Range llvm.amdgcn.workgroup.id.x [0,4)
// CHECK-THIRD: [DimSpec] Range llvm.amdgcn.workgroup.id.y [0,4)
// CHECK-THIRD: [DimSpec] Range llvm.amdgcn.workgroup.id.y [0,4)
// CHECK-THIRD: [DimSpec] Range llvm.amdgcn.workgroup.id.z [0,4)
// CHECK-THIRD: [DimSpec] Range llvm.amdgcn.workgroup.id.z [0,4)
// CHECK-FIRST: [LaunchBoundSpec] MaxThreads 64 MinBlocksPerSM 0
// CHECK-THIRD: [LaunchBoundSpec] MaxThreads 64 MinBlocksPerSM 0
// CHECK-SECOND: [ObjectCacheChain] Hit at level 0 (Storage) for hash {{[0-9]+}}
// CHECK-FOURTH: [ObjectCacheChain] Hit at level 0 (Storage) for hash {{[0-9]+}}
// CHECK: Block dim.x = 4
// CHECK: Block dim.y = 4
// CHECK: Block dim.z = 4
// CHECK: Grid dim.x = 4
// CHECK: Grid dim.y = 4
// CHECK: Grid dim.z = 4
// CHECK-FIRST: [proteus][JitEngineDevice] MemoryCache rank 0 hits 1 accesses 3
// CHECK-SECOND: [proteus][JitEngineDevice] MemoryCache rank 0 hits 1 accesses 3
// CHECK-FIRST: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 2 NumHits 1
// CHECK-FIRST: [proteus][JitEngineDevice] StorageCache rank 0 hits 0 accesses 2
// CHECK-SECOND: [proteus][JitEngineDevice] StorageCache rank 0 hits 2 accesses 2
// CHECK-FOURTH: [proteus][JitEngineDevice] StorageCache rank 0 hits 3 accesses 3
