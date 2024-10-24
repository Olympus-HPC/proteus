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
using namespace llvm::orc;

class JitEngineDeviceCUDA : public JitEngineDevice<JitEngineDeviceCUDA> {
public:
  static JitEngineDeviceCUDA &instance();

  ~JitEngineDeviceCUDA() {
    CodeCache.printStats("CUDA engine");
    StoredCache.printStats();
  }

  cudaError_t compileAndRun(StringRef ModuleUniqueId, StringRef KernelName,
                            FatbinWrapper_t *FatbinWrapper, size_t FatbinSize,
                            RuntimeConstant *RC, int NumRuntimeConstants,
                            dim3 GridDim, dim3 BlockDim, void **KernelArgs,
                            uint64_t ShmemSize, void *Stream);

  void *resolveDeviceGlobalAddr(const void *Addr);

  void setLaunchBoundsForKernel(Module *M, Function *F, int GridSize,
                                int BlockSize);

private:
  JitCache<CUfunction> CodeCache;
  JitStoredCache<CUfunction> StoredCache;
  std::string CudaArch;

  JitEngineDeviceCUDA();
  JitEngineDeviceCUDA(JitEngineDeviceCUDA &) = delete;
  JitEngineDeviceCUDA(JitEngineDeviceCUDA &&) = delete;

  std::unique_ptr<MemoryBuffer> extractDeviceBitcode(StringRef KernelName,
                                                     const char *Binary,
                                                     size_t FatbinSize = 0);

  cudaError_t codegenAndLaunch(Module *M, StringRef DeviceArch,
                               StringRef KernelName, StringRef Suffix,
                               uint64_t HashValue, RuntimeConstant *RC,
                               int NumRuntimeConstants, dim3 GridDim,
                               dim3 BlockDim, void **KernelArgs,
                               uint64_t ShmemSize, CUstream Stream);

  void codegenPTX(Module &M, StringRef DeviceArch,
                  SmallVectorImpl<char> &PTXStr);
};

} // namespace proteus

#endif
