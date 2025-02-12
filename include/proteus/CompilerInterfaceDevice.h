//===-- CompilerInterfaceDevice.h -- JIT entry point for GPU header --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_COMPILERINTERFACEDEVICE_H
#define PROTEUS_COMPILERINTERFACEDEVICE_H

#if PROTEUS_ENABLE_CUDA

#include "proteus/JitEngineDeviceCUDA.hpp"
using JitDeviceImplT = proteus::JitEngineDeviceCUDA;

#elif PROTEUS_ENABLE_HIP

#include "proteus/JitEngineDeviceHIP.hpp"
using JitDeviceImplT = proteus::JitEngineDeviceHIP;

#else
#error                                                                         \
    "CompilerInterfaceDevice requires PROTEUS_ENABLE_CUDA or PROTEUS_ENABLE_HIP"
#endif

// The ABI of __jit_launch_kernel mirrors device-specific launchKernel and
// depends on the host arch: https://github.com/Olympus-HPC/proteus/issues/47.
extern "C" proteus::DeviceTraits<JitDeviceImplT>::DeviceError_t
__jit_launch_kernel(void *Kernel, dim3 GridDim, dim3 BlockDim,
                    void **KernelArgs, uint64_t ShmemSize, void *Stream);

#endif
