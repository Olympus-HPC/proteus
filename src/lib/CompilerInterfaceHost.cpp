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
#include "proteus/JitEngineHost.hpp"
#include "proteus/LambdaRegistry.hpp"
#include "proteus/Utils.h"

using namespace proteus;
using namespace llvm;

extern "C" __attribute__((used)) void *__jit_entry(char *FnName, char *IR,
                                                   int IRSize,
                                                   RuntimeConstant *RC,
                                                   int NumRuntimeConstants) {
  TIMESCOPE("__jit_entry");
  JitEngineHost &Jit = JitEngineHost::instance();
#if PROTEUS_ENABLE_DEBUG
  Logger::logs("proteus") << "FnName " << FnName << " NumRuntimeConstants "
                          << NumRuntimeConstants << "\n";
  for (int I = 0; I < NumRuntimeConstants; ++I)
    Logger::logs("proteus")
        << " Value Int32=" << RC[I].Value.Int32Val
        << " Value Int64=" << RC[I].Value.Int64Val
        << " Value Float=" << RC[I].Value.FloatVal
        << " Value Double=" << RC[I].Value.DoubleVal << "\n";
#endif

  void *JitFnPtr =
      Jit.compileAndLink(FnName, IR, IRSize, RC, NumRuntimeConstants);

  return JitFnPtr;
}

extern "C" __attribute__((used)) void __jit_push_variable(RuntimeConstant RC) {
  LambdaRegistry::instance().pushJitVariable(RC);
}

extern "C" __attribute__((used)) void
__jit_register_lambda(const char *Symbol) {
  LambdaRegistry::instance().registerLambda(Symbol);
}
