//===-- JitEngineDeviceHIP.hpp -- JIT Engine Device for HIP header --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_JITENGINEDEVICEHIP_HPP
#define PROTEUS_JITENGINEDEVICEHIP_HPP

#include "JitEngineDevice.hpp"
#include "Utils.h"

namespace proteus {

using namespace llvm;

class JitEngineDeviceHIP;
template <> struct DeviceTraits<JitEngineDeviceHIP> {
  using DeviceError_t = hipError_t;
  using DeviceStream_t = hipStream_t;
  using KernelFunction_t = hipFunction_t;
};

class JitEngineDeviceHIP : public JitEngineDevice<JitEngineDeviceHIP> {
public:
  static JitEngineDeviceHIP &instance();

  void *resolveDeviceGlobalAddr(const void *Addr);

  void setLaunchBoundsForKernel(Module &M, Function &F, size_t GridSize,
                                int BlockSize);

  std::unique_ptr<MemoryBuffer> extractDeviceBitcode(StringRef KernelName,
                                                     void *Kernel);

  std::unique_ptr<MemoryBuffer> codegenObject(Module &M, StringRef DeviceArch);

  hipFunction_t getKernelFunctionFromImage(StringRef KernelName,
                                           const void *Image);

  hipError_t launchKernelFunction(hipFunction_t KernelFunc, dim3 GridDim,
                                  dim3 BlockDim, void **KernelArgs,
                                  uint64_t ShmemSize, hipStream_t Stream);

  hipError_t launchKernelDirect(void *KernelFunc, dim3 GridDim, dim3 BlockDim,
                                void **KernelArgs, uint64_t ShmemSize,
                                hipStream_t Stream);

private:
  JitEngineDeviceHIP();
  JitEngineDeviceHIP(JitEngineDeviceHIP &) = delete;
  JitEngineDeviceHIP(JitEngineDeviceHIP &&) = delete;
};

} // namespace proteus

#endif
