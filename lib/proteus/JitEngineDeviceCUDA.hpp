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

#include "proteus/JitEngineDevice.hpp"
#include "proteus/Utils.h"
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

  static const SmallVector<StringRef> gridDimXFnName() {
    return {"llvm.nvvm.read.ptx.sreg.nctaid.x"};
  };

  static const SmallVector<StringRef> gridDimYFnName() {
    return {"llvm.nvvm.read.ptx.sreg.nctaid.y"};
  };

  static const SmallVector<StringRef> gridDimZFnName() {
    return {"llvm.nvvm.read.ptx.sreg.nctaid.z"};
  };

  static const SmallVector<StringRef> blockDimXFnName() {
    return {"llvm.nvvm.read.ptx.sreg.ntid.x"};
  };

  static const SmallVector<StringRef> blockDimYFnName() {
    return {"llvm.nvvm.read.ptx.sreg.ntid.y"};
  };

  static const SmallVector<StringRef> blockDimZFnName() {
    return {"llvm.nvvm.read.ptx.sreg.ntid.z"};
  };

  static const SmallVector<StringRef> blockIdxXFnName() {
    return {"llvm.nvvm.read.ptx.sreg.ctaid.x"};
  };

  static const SmallVector<StringRef> blockIdxYFnName() {
    return {"llvm.nvvm.read.ptx.sreg.ctaid.y"};
  };

  static const SmallVector<StringRef> blockIdxZFnName() {
    return {"llvm.nvvm.read.ptx.sreg.ctaid.z"};
  };

  static const SmallVector<StringRef> threadIdxXFnName() {
    return {"llvm.nvvm.read.ptx.sreg.tid.x"};
  };

  static const SmallVector<StringRef> threadIdxYFnName() {
    return {"llvm.nvvm.read.ptx.sreg.tid.y"};
  };

  static const SmallVector<StringRef> threadIdxZFnName() {
    return {"llvm.nvvm.read.ptx.sreg.tid.z"};
  };

  static bool isHashedSection(StringRef sectionName) {
    static const std::string Section{".nv_fatbin"};
    return Section.compare(sectionName) == 0;
  }

  void *resolveDeviceGlobalAddr(const void *Addr);

  static void setLaunchBoundsForKernel(Module &M, Function &F, size_t GridSize,
                                       int BlockSize);

  Module &extractDeviceBitcode(StringRef KernelName, void *Kernel);

  void codegenPTX(Module &M, StringRef DeviceArch,
                  SmallVectorImpl<char> &PTXStr);

  static std::unique_ptr<MemoryBuffer>
  codegenObject(Module &M, StringRef DeviceArch, bool UseCUDArtc = false);

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
