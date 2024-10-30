//===-- JitEngineDeviceCUDA.hpp -- JIT Engine Device for CUDA header --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_JITENGINEDEVICECUDA_HPP
#define PROTEUS_JITENGINEDEVICECUDA_HPP

#include "JitEngineDevice.hpp"
#include "Utils.h"

namespace proteus {

using namespace llvm;

class JitEngineDeviceCUDA;
template <> struct DeviceTraits<JitEngineDeviceCUDA> {
  using DeviceError_t = cudaError_t;
  using DeviceStream_t = CUstream;
  using KernelFunction_t = CUfunction;
};

class JitEngineDeviceCUDA : public JitEngineDevice<JitEngineDeviceCUDA> {
public:
  static JitEngineDeviceCUDA &instance();

protected:
  friend JitEngineDevice<JitEngineDeviceCUDA>;

  void *resolveDeviceGlobalAddr(const void *Addr);

  void setLaunchBoundsForKernel(Module *M, Function *F, int GridSize,
                                int BlockSize);

  std::unique_ptr<MemoryBuffer> extractDeviceBitcode(StringRef KernelName,
                                                     const char *Binary,
                                                     size_t FatbinSize = 0);

  void codegenPTX(Module &M, StringRef DeviceArch,
                  SmallVectorImpl<char> &PTXStr);

  std::unique_ptr<MemoryBuffer> codegenObject(Module &M, StringRef DeviceArch);

  cudaError_t
  cudaModuleLaunchKernel(CUfunction f, unsigned int gridDimX,
                         unsigned int gridDimY, unsigned int gridDimZ,
                         unsigned int blockDimX, unsigned int blockDimY,
                         unsigned int blockDimZ, unsigned int sharedMemBytes,
                         CUstream hStream, void **kernelParams, void **extra);

  CUfunction getKernelFunctionFromImage(StringRef KernelName,
                                        const void *Image);

  cudaError_t launchKernelFunction(CUfunction KernelFunc, dim3 GridDim,
                                   dim3 BlockDim, void **KernelArgs,
                                   uint64_t ShmemSize, CUstream Stream);

private:
  JitEngineDeviceCUDA();
  JitEngineDeviceCUDA(JitEngineDeviceCUDA &) = delete;
  JitEngineDeviceCUDA(JitEngineDeviceCUDA &&) = delete;
};

} // namespace proteus

#endif
