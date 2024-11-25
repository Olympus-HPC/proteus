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

#include "CompilerInterfaceTypes.h"
#include "JitEngineDevice.hpp"
#include <llvm/ADT/ArrayRef.h>
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
    const char *ModuleUniqueId, void* Kernel,
    proteus::FatbinWrapper_t *FatbinWrapper, size_t FatbinSize,
    dim3 GridDim, dim3 BlockDim,
    void **KernelArgs, uint64_t ShmemSize, void *Stream) {

  using namespace llvm;
  using namespace proteus;
  auto &Jit = JitDeviceImplT::instance();
  auto optionalKernelInfo = Jit.getJITKernelInfo(Kernel);
  if (!optionalKernelInfo) {
    #if ENABLE_CUDA
      return cudaLaunchKernel((const void*)Kernel, GridDim, BlockDim, KernelArgs, ShmemSize,
        static_cast<typename JitDeviceImplT::DeviceStream_t>(Stream));
    #elif ENABLE_HIP
      return hipLaunchKernel((const void*)Kernel, GridDim, BlockDim, KernelArgs, ShmemSize,
        static_cast<typename JitDeviceImplT::DeviceStream_t>(Stream));
    #endif
  }

  const auto& KernelInfo = optionalKernelInfo.value();
  const char* KernelName = KernelInfo.getName();
  int32_t NumRuntimeConstants = KernelInfo.getNumRCs();
  auto RCIndices = KernelInfo.getRCIndices();

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

  return Jit.compileAndRun(
      ModuleUniqueId, KernelName, FatbinWrapper, FatbinSize, RCIndices,
      NumRuntimeConstants, GridDim, BlockDim, KernelArgs, ShmemSize,
      static_cast<typename JitDeviceImplT::DeviceStream_t>(Stream));
}

#endif