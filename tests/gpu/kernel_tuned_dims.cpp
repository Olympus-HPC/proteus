// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_TUNED_KERNELS=%S/tuned_dims.json PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization;cache-stats" %build/kernel_tuned_dims.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// RUN: PROTEUS_TUNED_KERNELS=%S/tuned_dims.json PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization;cache-stats" %build/kernel_tuned_dims.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

extern "C" __global__ __attribute__((annotate("jit"))) void tuned_dims() {
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 &&
      blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)
    printf("Observed GridDim: (%u,%u,%u) BlockDim: (%u,%u,%u)\n", gridDim.x,
           gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);
}

int main() {
  tuned_dims<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>();
  gpuErrCheck(gpuDeviceSynchronize());

  tuned_dims<<<dim3(4, 1, 1), dim3(2, 1, 1)>>>();
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// clang-format off
// CHECK-FIRST: [KernelConfig] ID:tuned_dims {{.*}}GridDim:(2,3,1) BlockDim:(4,2,1)
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI13__HIP_GridDimE3__XcvjEv|_ZL20__hip_get_grid_dim_xv|llvm.nvvm.read.ptx.sreg.nctaid.x}} with constant i32 2
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI13__HIP_GridDimE3__YcvjEv|_ZL20__hip_get_grid_dim_yv|llvm.nvvm.read.ptx.sreg.nctaid.y}} with constant i32 3
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI13__HIP_GridDimE3__ZcvjEv|_ZL20__hip_get_grid_dim_zv|llvm.nvvm.read.ptx.sreg.nctaid.z}} with constant i32 1
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI14__HIP_BlockDimE3__XcvjEv|_ZL21__hip_get_block_dim_xv|llvm.nvvm.read.ptx.sreg.ntid.x}} with constant i32 4
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI14__HIP_BlockDimE3__YcvjEv|_ZL21__hip_get_block_dim_yv|llvm.nvvm.read.ptx.sreg.ntid.y}} with constant i32 2
// CHECK-FIRST: [DimSpec] Replace call to {{_ZNK17__HIP_CoordinatesI14__HIP_BlockDimE3__ZcvjEv|_ZL21__hip_get_block_dim_zv|llvm.nvvm.read.ptx.sreg.ntid.z}} with constant i32 1
// CHECK-FIRST: [LaunchBoundSpec] MaxThreads 8 MinBlocksPerSM 2
// CHECK: Observed GridDim: (2,3,1) BlockDim: (4,2,1)
// CHECK: Observed GridDim: (2,3,1) BlockDim: (4,2,1)
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 hits 1 accesses 2
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 2 NumHits 1
// CHECK-FIRST: [proteus][JitEngineDevice] StorageCache rank 0 hits 0 accesses 1
// CHECK-SECOND: [proteus][JitEngineDevice] StorageCache rank 0 hits 1 accesses 1
// clang-format on
