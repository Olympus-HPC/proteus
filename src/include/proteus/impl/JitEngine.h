//===-- JitEngine.h -- Base JIT Engine header --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_JITENGINE_H
#define PROTEUS_JITENGINE_H

#include <cstdlib>
#include <string>

#include <llvm/ADT/DenseMap.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/Module.h>
#include <llvm/Target/TargetMachine.h>

#include "proteus/impl/CompilerInterfaceRuntimeConstantInfo.h"
#include "proteus/impl/Config.h"
#include "proteus/impl/CoreLLVM.h"
#include "proteus/Error.h"
#include "proteus/impl/Hashing.h"

namespace proteus {

using namespace llvm;

class JitEngine {
public:
  InitLLVMTargets Init;
  bool isProteusDisabled() { return Config::get().ProteusDisable; }

  void enable() { Config::get().ProteusDisable = false; }

  void disable() { Config::get().ProteusDisable = true; }

protected:
  SmallVector<RuntimeConstant>
  getRuntimeConstantValues(void **KernelArgs,
                           ArrayRef<RuntimeConstantInfo *> RCInfoArray);

  JitEngine();
};

} // namespace proteus

#endif
