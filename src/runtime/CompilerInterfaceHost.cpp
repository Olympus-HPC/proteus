//===-- CompilerInterfaceHost.cpp -- JIT library entry point for CPU --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/impl/Caching/ObjectCacheRegistry.h"
#include "proteus/impl/CompilerInterfaceRuntimeConstantInfo.h"
#include "proteus/impl/JitEngineHost.h"
#include "proteus/impl/LambdaRegistry.h"

using namespace proteus;
using namespace llvm;

// NOLINTBEGIN(readability-identifier-naming)

extern "C" __attribute__((used)) void *
__jit_entry(char *FnName, char *IR, int IRSize, void **Args,
            RuntimeConstantInfo **RCInfoArrayPtr, int NumRuntimeConstants) {
  TIMESCOPE("__jit_entry");
  JitEngineHost &Jit = JitEngineHost::instance();
  ArrayRef<RuntimeConstantInfo *> RCInfoArray{
      RCInfoArrayPtr, static_cast<size_t>(NumRuntimeConstants)};
  void *JitFnPtr = Jit.compileAndLink(FnName, IR, IRSize, Args, RCInfoArray);

  return JitFnPtr;
}

extern "C" __attribute__((used)) void __jit_push_variable(RuntimeConstant RC) {
  LambdaRegistry::instance().pushJitVariable(RC);
}

extern "C" __attribute__((used)) void
__jit_register_lambda(const char *Symbol) {
  LambdaRegistry::instance().registerLambda(Symbol);
}

extern "C" void __jit_enable_host() {
  JitEngineHost &Jit = JitEngineHost::instance();
  Jit.enable();
}

extern "C" void __jit_disable_host() {
  JitEngineHost &Jit = JitEngineHost::instance();
  Jit.disable();
}

// NOLINTEND(readability-identifier-naming)
