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

  static const char *gridDimXFnName() {
    return "_ZNK17__HIP_CoordinatesI13__HIP_GridDimE3__XcvjEv";
  };

  static const char *gridDimYFnName() {
    return "_ZNK17__HIP_CoordinatesI13__HIP_GridDimE3__YcvjEv";
  };

  static const char *gridDimZFnName() {
    return "_ZNK17__HIP_CoordinatesI13__HIP_GridDimE3__ZcvjEv";
  };

  static const char *blockDimXFnName() {
    return "_ZNK17__HIP_CoordinatesI14__HIP_BlockDimE3__XcvjEv";
  };

  static const char *blockDimYFnName() {
    return "_ZNK17__HIP_CoordinatesI14__HIP_BlockDimE3__YcvjEv";
  };

  static const char *blockDimZFnName() {
    return "_ZNK17__HIP_CoordinatesI14__HIP_BlockDimE3__ZcvjEv";
  };

  static const char *blockIdxXFnName() {
    return "_ZNK17__HIP_CoordinatesI15__HIP_ThreadIdxE3__XcvjEv";
  };

  static const char *blockIdxYFnName() {
    return "_ZNK17__HIP_CoordinatesI15__HIP_ThreadIdxE3__YcvjEv";
  };

  static const char *blockIdxZFnName() {
    return "_ZNK17__HIP_CoordinatesI15__HIP_ThreadIdxE3__ZcvjEv";
  };

  static const char *threadIdxXFnName() {
    return "_ZNK17__HIP_CoordinatesI14__HIP_BlockIdxE3__XcvjEv";
  };

  static const char *threadIdxYFnName() {
    return "_ZNK17__HIP_CoordinatesI14__HIP_BlockIdxE3__YcvjEv";
  };

  static const char *threadIdxZFnName() {
    return "_ZNK17__HIP_CoordinatesI14__HIP_BlockIdxE3__ZcvjEv";
  };

  static bool HashSection(StringRef sectionName) {
    static const std::string Section{".hip_fatbin"};
    return Section.compare(sectionName) == 0;
  }

  void *resolveDeviceGlobalAddr(const void *Addr);

  void setLaunchBoundsForKernel(Module &M, Function &F, size_t GridSize,
                                int BlockSize);

  void setKernelDims(Module &M, dim3 &GridDim, dim3 &BlockDim);

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
