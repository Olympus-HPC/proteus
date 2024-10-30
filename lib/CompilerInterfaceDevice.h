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

#include "JitEngineDevice.hpp"
#if ENABLE_CUDA
#include "JitEngineDeviceCUDA.hpp"
using JitDeviceImplT = proteus::JitEngineDeviceCUDA;
#elif ENABLE_HIP
#include "JitEngineDeviceHIP.hpp"
using JitDeviceImplT = proteus::JitEngineDeviceHIP;
#else
#error "CompilerInterfaceDevice requires ENABLE_CUDA or ENABLE_HIP"
#endif

// Return "auto" should resolve to cudaError_t or hipError_t.
static inline auto __jit_launch_kernel_internal(
    const char *ModuleUniqueId, char *KernelName,
    proteus::FatbinWrapper_t *FatbinWrapper, size_t FatbinSize,
    RuntimeConstant *RC, int NumRuntimeConstants, dim3 GridDim, dim3 BlockDim,
    void **KernelArgs, uint64_t ShmemSize, void *Stream) {

  using namespace llvm;
  using namespace proteus;

  auto printKernelLaunchInfo = [&]() {
    dbgs() << "JIT Launch Kernel\n";
    dbgs() << "=== Kernel Info\n";
    dbgs() << "KernelName " << KernelName << "\n";
    dbgs() << "FatbinSize " << FatbinSize << "\n";
    dbgs() << "Grid " << GridDim.x << ", " << GridDim.y << ", " << GridDim.z
           << "\n";
    dbgs() << "Block " << BlockDim.x << ", " << BlockDim.y << ", " << BlockDim.z
           << "\n";
    dbgs() << "KernelArgs " << KernelArgs << "\n";
    dbgs() << "ShmemSize " << ShmemSize << "\n";
    dbgs() << "Stream " << Stream << "\n";
    dbgs() << "=== End Kernel Info\n";
  };

  TIMESCOPE("__jit_launch_kernel");
  DBG(printKernelLaunchInfo());
  auto &Jit = JitDeviceImplT::instance();
  return Jit.compileAndRun(
      ModuleUniqueId, KernelName, FatbinWrapper, FatbinSize, RC,
      NumRuntimeConstants, GridDim, BlockDim, KernelArgs, ShmemSize,
      static_cast<typename JitDeviceImplT::DeviceStream_t>(Stream));
}

#endif