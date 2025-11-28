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
#include <memory>

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

  void setLaunchBoundsForKernel(Module &M, Function &F, size_t GridSize,
                                int BlockSize);

  void extractModules(BinaryInfo &BinInfo);

  std::unique_ptr<Module> tryExtractKernelModule(BinaryInfo &BinInfo,
                                                 StringRef KernelName,
                                                 LLVMContext &Ctx);

  void codegenPTX(Module &M, StringRef DeviceArch,
                  SmallVectorImpl<char> &PTXStr);

  HashT getModuleHash(BinaryInfo &BinInfo);

  std::unique_ptr<MemoryBuffer> compileOnly(Module &M,
                                            bool DisableIROpt = false);

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
