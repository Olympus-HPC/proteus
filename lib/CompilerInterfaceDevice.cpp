//===-- CompilerInterfaceDevice.cpp -- JIT library entry point for GPU --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "CompilerInterfaceDevice.h"

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
