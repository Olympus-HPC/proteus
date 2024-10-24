//===-- JitEngineDeviceHIP.hpp -- JIT library entry point for GPU --===//
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
using namespace llvm::orc;

class JitEngineDeviceHIP : public JitEngineDevice<JitEngineDeviceHIP> {
public:
  static JitEngineDeviceHIP &instance();

  ~JitEngineDeviceHIP() {
    CodeCache.printStats("HIP engine");
    StoredCache.printStats();
  }

  hipError_t compileAndRun(StringRef ModuleUniqueId, StringRef KernelName,
                           FatbinWrapper_t *FatbinWrapper, size_t FatbinSize,
                           RuntimeConstant *RC, int NumRuntimeConstants,
                           dim3 GridDim, dim3 BlockDim, void **KernelArgs,
                           uint64_t ShmemSize, void *Stream);

  void *resolveDeviceGlobalAddr(const void *Addr);

  void setLaunchBoundsForKernel(Module *M, Function *F, int GridSize,
                                int BlockSize);

private:
  JitCache<hipFunction_t> CodeCache;
  JitStoredCache<hipFunction_t> StoredCache;

  JitEngineDeviceHIP();
  JitEngineDeviceHIP(JitEngineDeviceHIP &) = delete;
  JitEngineDeviceHIP(JitEngineDeviceHIP &&) = delete;

  std::unique_ptr<MemoryBuffer> extractDeviceBitcode(StringRef KernelName,
                                                     const char *Binary,
                                                     size_t FatbinSize = 0);

  hipError_t codegenAndLaunch(Module *M, StringRef DeviceArch,
                              StringRef KernelName, StringRef Suffix,
                              uint64_t HashValue, RuntimeConstant *RC,
                              int NumRuntimeConstants, dim3 GridDim,
                              dim3 BlockDim, void **KernelArgs,
                              uint64_t ShmemSize, hipStream_t Stream);
};

} // namespace proteus

#endif
