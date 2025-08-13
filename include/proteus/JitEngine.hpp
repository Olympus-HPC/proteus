//===-- JitEngine.hpp -- Base JIT Engine header --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_JITENGINE_HPP
#define PROTEUS_JITENGINE_HPP

#include <cstdlib>
#include <optional>
#include <string>

#include <llvm/ADT/DenseMap.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/Module.h>
#include <llvm/Target/TargetMachine.h>

#include "proteus/CompilerInterfaceRuntimeConstantInfo.h"
#include "proteus/Config.hpp"
#include "proteus/CoreLLVM.hpp"
#include "proteus/Debug.h"
#include "proteus/Hashing.hpp"
#include "proteus/Logger.hpp"

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

  std::string mangleSuffix(HashT &HashValue);
};

} // namespace proteus

#endif
