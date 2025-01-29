//===-- CompilerInterfaceDeviceCUDA.cpp -- JIT entry point for CUDA GPU --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "proteus/CompilerInterfaceDevice.h"
#include "proteus/CompilerInterfaceDeviceInternal.hpp"

using namespace proteus;

extern "C" __attribute__((used)) cudaError_t
__jit_launch_kernel(const char *ModuleUniqueId, void *Kernel, dim3 GridDim,
                    dim3 BlockDim, void **KernelArgs, uint64_t ShmemSize,
                    void *Stream) {
  return __jit_launch_kernel_internal(ModuleUniqueId, Kernel, GridDim, BlockDim,
                                      KernelArgs, ShmemSize, Stream);
}
