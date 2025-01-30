//===-- CompilerInterfaceDeviceHIP.cpp -- JIT entry point for HIP GPU --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "CompilerInterfaceDevice.h"
#include "CompilerInterfaceDeviceInternal.hpp"

// NOTE: Using the ABI With scalars for GridDim, BlockDim instead of dim3 to
// avoid issues with aggregate coercion of parameters. Caller packs XY in a
// uint64_t.
extern "C" __attribute__((used)) hipError_t __jit_launch_kernel(
    void *Kernel, uint64_t GridDimXY, uint32_t GridDimZ, uint64_t BlockDim_XY,
    uint32_t BlockDimZ, void **KernelArgs, uint64_t ShmemSize, void *Stream) {
  dim3 GridDim = {*(uint32_t *)&GridDimXY, *(((uint32_t *)&GridDimXY) + 1),
                  GridDimZ};
  dim3 BlockDim = {*(uint32_t *)&BlockDim_XY, *(((uint32_t *)&BlockDim_XY) + 1),
                   BlockDimZ};

  return __jit_launch_kernel_internal(Kernel, GridDim, BlockDim, KernelArgs,
                                      ShmemSize, Stream);
}
