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
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>

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

  static const char *gridDimXFnName() {
    return "llvm.nvvm.read.ptx.sreg.nctaid.x";
  };

  static const char *gridDimYFnName() {
    return "llvm.nvvm.read.ptx.sreg.nctaid.y";
  };

  static const char *gridDimZFnName() {
    return "llvm.nvvm.read.ptx.sreg.nctaid.z";
  };

  static const char *blockDimXFnName() {
    return "llvm.nvvm.read.ptx.sreg.ntid.x";
  };

  static const char *blockDimYFnName() {
    return "llvm.nvvm.read.ptx.sreg.ntid.y";
  };

  static const char *blockDimZFnName() {
    return "llvm.nvvm.read.ptx.sreg.ntid.z";
  };

  static const char *blockIdxXFnName() {
    return "llvm.nvvm.read.ptx.sreg.ctaid.x";
  };

  static const char *blockIdxYFnName() {
    return "llvm.nvvm.read.ptx.sreg.ctaid.y";
  };

  static const char *blockIdxZFnName() {
    return "llvm.nvvm.read.ptx.sreg.ctaid.z";
  };

  static const char *threadIdxXFnName() {
    return "llvm.nvvm.read.ptx.sreg.tid.x";
  };

  static const char *threadIdxYFnName() {
    return "llvm.nvvm.read.ptx.sreg.tid.y";
  };

  static const char *threadIdxZFnName() {
    return "llvm.nvvm.read.ptx.sreg.tid.z";
  };

  static const char *getFatBinSectionName() { return ".nv_fatbin"; }

  void *resolveDeviceGlobalAddr(const void *Addr);

  void setLaunchBoundsForKernel(Module &M, Function &F, size_t GridSize,
                                int BlockSize);

  int extractDeviceBitcode(StringRef KernelName, void *Kernel);

  std::unique_ptr<llvm::Module>
  createLinkedModule(ArrayRef<std::unique_ptr<llvm::Module>> LinkedModules,
                     StringRef KernelName);

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

  cudaError_t launchKernelDirect(void *KernelFunc, dim3 GridDim, dim3 BlockDim,
                                 void **KernelArgs, uint64_t ShmemSize,
                                 CUstream Stream);

private:
  JitEngineDeviceCUDA();
  JitEngineDeviceCUDA(JitEngineDeviceCUDA &) = delete;
  JitEngineDeviceCUDA(JitEngineDeviceCUDA &&) = delete;

  void extractLinkedBitcode(LLVMContext &Ctx, CUmodule &CUMod,
                            SmallVector<std::unique_ptr<Module>> &LinkedModules,
                            std::string &ModuleId);
};

} // namespace proteus

#endif
