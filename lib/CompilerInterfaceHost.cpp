//===-- CompilerInterfaceHost.cpp -- JIT library entry point for CPU --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "CompilerInterfaceTypes.h"
#include "JitEngineHost.hpp"
#include "Utils.h"

using namespace proteus;
using namespace llvm;

extern "C" __attribute__((used)) void *__jit_entry(char *FnName, char *IR,
                                                   int IRSize,
                                                   RuntimeConstant *RC,
                                                   int NumRuntimeConstants) {
  TIMESCOPE("__jit_entry");
  JitEngineHost &Jit = JitEngineHost::instance();
#if ENABLE_DEBUG
  dbgs() << "FnName " << FnName << " NumRuntimeConstants "
         << NumRuntimeConstants << "\n";
  for (int I = 0; I < NumRuntimeConstants; ++I)
    dbgs() << " Value Int32=" << RC[I].Int32Val
           << " Value Int64=" << RC[I].Int64Val
           << " Value Float=" << RC[I].FloatVal
           << " Value Double=" << RC[I].DoubleVal << "\n";
#endif

  void *JitFnPtr =
      Jit.compileAndLink(FnName, IR, IRSize, RC, NumRuntimeConstants);

  return JitFnPtr;
}
