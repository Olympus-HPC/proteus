//===-- CompilerInterfaceDevice.cpp -- JIT library entry point for GPU --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "proteus/impl/CompilerInterfaceDevice.h"
#include "proteus/impl/CompilerInterfaceDeviceInternal.h"
#include "proteus/impl/JitEngineDevice.h"
#include "proteus/impl/JitEngineInfoRegistry.h"

using namespace proteus;

// NOLINTBEGIN(readability-identifier-naming)

// NOTE: A great mystery is: why does this work ONLY if HostAddr is a CONST
// void* for HIP
extern "C" __attribute((used)) void __jit_register_var(void *Handle,
                                                       const void *HostAddr,
                                                       const char *VarName,
                                                       uint64_t VarSize) {
  auto &JitEngineInfo = JitEngineInfoRegistry::instance();
  // NOTE: For HIP it works to get the symobl address during the call inside a
  // constructor context, but for CUDA, it fails.  So we save the host address
  // and defer resolving the symbol address when patching the bitcode, which
  // works for both CUDA and HIP.
  JitEngineInfo.registerVar(Handle, HostAddr, VarName, VarSize);
}

extern "C" __attribute__((used)) void
__jit_register_fatbinary(void *Handle, void *FatbinWrapper,
                         const char *ModuleId) {
  auto &JitEngineInfo = JitEngineInfoRegistry::instance();
  JitEngineInfo.registerFatBinary(Handle, FatbinWrapper, ModuleId);
}

extern "C" __attribute__((used)) void
__jit_register_fatbinary_end(void *Handle) {
  auto &JitEngineInfo = JitEngineInfoRegistry::instance();
  JitEngineInfo.registerFatBinaryEnd(Handle);
}

extern "C" __attribute__((used)) void
__jit_register_linked_binary(void *FatbinWrapper, const char *ModuleId) {
  auto &JitEngineInfo = JitEngineInfoRegistry::instance();
  JitEngineInfo.registerLinkedBinary(FatbinWrapper, ModuleId);
}

extern "C" __attribute((used)) void
__jit_register_function(void *Handle, void *Kernel, char *KernelName,
                        RuntimeConstantInfo **RCInfoArrayPtr, int32_t NumRCs) {
  ArrayRef<RuntimeConstantInfo *> RCInfoArray{RCInfoArrayPtr,
                                              static_cast<size_t>(NumRCs)};
  auto &JitEngineInfo = JitEngineInfoRegistry::instance();
  JitEngineInfo.registerFunction(Handle, Kernel, KernelName, RCInfoArray);
}

extern "C" proteus::DeviceTraits<JitDeviceImplT>::DeviceError_t
__jit_launch_kernel(void *Kernel, dim3 GridDim, dim3 BlockDim,
                    void **KernelArgs, uint64_t ShmemSize, void *Stream) {
  return __jit_launch_kernel_internal(Kernel, GridDim, BlockDim, KernelArgs,
                                      ShmemSize, Stream);
}

extern "C" void __jit_enable_device() {
  auto &Jit = JitDeviceImplT::instance();
  Jit.enable();
}

extern "C" void __jit_disable_device() {
  auto &Jit = JitDeviceImplT::instance();
  Jit.disable();
}

// NOLINTEND(readability-identifier-naming)
