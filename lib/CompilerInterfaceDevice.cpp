//===-- CompilerInterfaceDevice.cpp -- JIT library entry point for GPU --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "proteus/CompilerInterfaceDevice.h"
#include "proteus/JitEngineDevice.hpp"

using namespace proteus;

// NOTE: A great mystery is: why does this work ONLY if HostAddr is a CONST
// void* for HIP
extern "C" __attribute((used)) void __jit_register_var(const void *HostAddr,
                                                       const char *VarName) {
  auto &Jit = JitDeviceImplT::instance();
  // NOTE: For HIP it works to get the symobl address during the call inside a
  // constructor context, but for CUDA, it fails.  So we save the host address
  // and defer resolving the symbol address when patching the bitcode, which
  // works for both CUDA and HIP.
  Jit.insertRegisterVar(VarName, HostAddr);
}

extern "C" __attribute__((used)) void
__jit_register_fatbinary(void *Handle, void *FatbinWrapper,
                         const char *ModuleId) {
  auto &Jit = JitDeviceImplT::instance();
  Jit.registerFatBinary(
      Handle, reinterpret_cast<FatbinWrapper_t *>(FatbinWrapper), ModuleId);
}

extern "C" __attribute__((used)) void __jit_register_fatbinary_end(void *) {
  auto &Jit = JitDeviceImplT::instance();
  Jit.registerFatBinaryEnd();
}

extern "C" __attribute__((used)) void
__jit_register_linked_binary(void *FatbinWrapper, const char *ModuleId) {
  auto &Jit = JitDeviceImplT::instance();
  Jit.registerLinkedBinary(reinterpret_cast<FatbinWrapper_t *>(FatbinWrapper),
                           ModuleId);
}

extern "C" __attribute((used)) void
__jit_register_function(void *Handle, void *Kernel, char *KernelName,
                        int32_t *RCIndices, int32_t *RCTypes, int32_t NumRCs) {
  auto &Jit = JitDeviceImplT::instance();
  Jit.registerFunction(Handle, Kernel, KernelName, RCIndices, RCTypes, NumRCs);
}
