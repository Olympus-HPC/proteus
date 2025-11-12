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

#include "proteus/JitEngineDevice.hpp"
#include "proteus/Utils.h"

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

  static const SmallVector<StringRef> gridDimXFnName() {
    return {"_ZNK17__HIP_CoordinatesI13__HIP_GridDimE3__XcvjEv",
            "llvm.amdgcn.num.workgroups.x"};
  };

  static const SmallVector<StringRef> gridDimYFnName() {
    return {"_ZNK17__HIP_CoordinatesI13__HIP_GridDimE3__YcvjEv",
            "llvm.amdgcn.num.workgroups.y"};
  };

  static const SmallVector<StringRef> gridDimZFnName() {
    return {"_ZNK17__HIP_CoordinatesI13__HIP_GridDimE3__ZcvjEv",
            "llvm.amdgcn.num.workgroups.z"};
  };

  static const SmallVector<StringRef> blockDimXFnName() {
    return {"_ZNK17__HIP_CoordinatesI14__HIP_BlockDimE3__XcvjEv",
            "llvm.amdgcn.workgroup.size.x"};
  };

  static const SmallVector<StringRef> blockDimYFnName() {
    return {"_ZNK17__HIP_CoordinatesI14__HIP_BlockDimE3__YcvjEv",
            "llvm.amdgcn.workgroup.size.y"};
  }

  static const SmallVector<StringRef> blockDimZFnName() {
    return {"_ZNK17__HIP_CoordinatesI14__HIP_BlockDimE3__ZcvjEv",
            "llvm.amdgcn.workgroup.size.z"};
  };

  static const SmallVector<StringRef> blockIdxXFnName() {
    return {"_ZNK17__HIP_CoordinatesI14__HIP_BlockIdxE3__XcvjEv",
            "llvm.amdgcn.workgroup.id.x"};
  };

  static const SmallVector<StringRef> blockIdxYFnName() {
    return {"_ZNK17__HIP_CoordinatesI14__HIP_BlockIdxE3__YcvjEv",
            "llvm.amdgcn.workgroup.id.y"};
  };

  static const SmallVector<StringRef> blockIdxZFnName() {
    return {"_ZNK17__HIP_CoordinatesI14__HIP_BlockIdxE3__ZcvjEv",
            "llvm.amdgcn.workgroup.id.z"};
  }

  static const SmallVector<StringRef> threadIdxXFnName() {
    return {"_ZNK17__HIP_CoordinatesI15__HIP_ThreadIdxE3__XcvjEv",
            "llvm.amdgcn.workitem.id.x"};
  };

  static const SmallVector<StringRef> threadIdxYFnName() {
    return {"_ZNK17__HIP_CoordinatesI15__HIP_ThreadIdxE3__YcvjEv",
            "llvm.amdgcn.workitem.id.y"};
  };

  static const SmallVector<StringRef> threadIdxZFnName() {
    return {"_ZNK17__HIP_CoordinatesI15__HIP_ThreadIdxE3__ZcvjEv",
            "llvm.amdgcn.workitem.id.z"};
  };

  void *resolveDeviceGlobalAddr(const void *Addr);

  void setKernelDims(Module &M, dim3 &GridDim, dim3 &BlockDim);

  void extractModules(BinaryInfo &BinInfo);

  std::unique_ptr<Module> tryExtractKernelModule(BinaryInfo &BinInfo,
                                                 StringRef KernelName,
                                                 LLVMContext &Ctx);

  hipFunction_t getKernelFunctionFromImage(
      StringRef KernelName, const void *Image,
      std::unordered_map<std::string, const void *> &VarNameToDevPtr);

  hipError_t launchKernelFunction(hipFunction_t KernelFunc, dim3 GridDim,
                                  dim3 BlockDim, void **KernelArgs,
                                  uint64_t ShmemSize, hipStream_t Stream);

  HashT getModuleHash(BinaryInfo &BinInfo);

  std::unique_ptr<MemoryBuffer> compileOnly(Module &M,
                                            bool DisableIROpt = false);

private:
  JitEngineDeviceHIP();
  JitEngineDeviceHIP(JitEngineDeviceHIP &) = delete;
  JitEngineDeviceHIP(JitEngineDeviceHIP &&) = delete;
};

} // namespace proteus

#endif
