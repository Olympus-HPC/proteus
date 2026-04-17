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
#include "proteus/TimeTracing.h"
#include "proteus/impl/CompilerInterfaceRuntimeConstantInfo.h"
#include "proteus/impl/JitEngineHost.h"
#include "proteus/impl/LambdaRegistry.h"

using namespace proteus;
using namespace llvm;

// NOLINTBEGIN(readability-identifier-naming)

extern "C" __attribute__((used)) void *
__proteus_entry(char *FnName, char *IR, int IRSize, void **Args,
                RuntimeConstantInfo **RCInfoArrayPtr, int NumRuntimeConstants) {
  TIMESCOPE("__proteus_entry");
  JitEngineHost &Jit = JitEngineHost::instance();
  ArrayRef<RuntimeConstantInfo *> RCInfoArray{
      RCInfoArrayPtr, static_cast<size_t>(NumRuntimeConstants)};
  void *JitFnPtr = Jit.compileAndLink(FnName, IR, IRSize, Args, RCInfoArray);

  return JitFnPtr;
}

extern "C" __attribute__((used)) void
__proteus_register_lambda_runtime_constant(int32_t Type, int32_t Pos,
                                           int32_t Offset, const void *ValuePtr,
                                           uint64_t ID) {
  RuntimeConstant RC{static_cast<RuntimeConstantType>(Type), Pos, Offset};
  // ValuePtr points at the lambda field storage (e.g., an i32 capture). It is
  // not a pointer to a full RuntimeConstantValue, so only copy the bytes
  // corresponding to the scalar type to avoid reading past the field (which
  // makes hashing/caching nondeterministic across runs).
  switch (static_cast<RuntimeConstantType>(Type)) {
  case RuntimeConstantType::BOOL:
    std::memcpy(&RC.Value.BoolVal, ValuePtr, sizeof(RC.Value.BoolVal));
    break;
  case RuntimeConstantType::INT8:
    std::memcpy(&RC.Value.Int8Val, ValuePtr, sizeof(RC.Value.Int8Val));
    break;
  case RuntimeConstantType::INT32:
    std::memcpy(&RC.Value.Int32Val, ValuePtr, sizeof(RC.Value.Int32Val));
    break;
  case RuntimeConstantType::INT64:
    std::memcpy(&RC.Value.Int64Val, ValuePtr, sizeof(RC.Value.Int64Val));
    break;
  case RuntimeConstantType::FLOAT:
    std::memcpy(&RC.Value.FloatVal, ValuePtr, sizeof(RC.Value.FloatVal));
    break;
  case RuntimeConstantType::DOUBLE:
    std::memcpy(&RC.Value.DoubleVal, ValuePtr, sizeof(RC.Value.DoubleVal));
    break;
  case RuntimeConstantType::PTR:
    std::memcpy(&RC.Value.PtrVal, ValuePtr, sizeof(RC.Value.PtrVal));
    break;
  default:
    reportFatalError(
        "__proteus_push_lambda_runtime_constant only supports scalar captures");
  }
  LambdaRegistry::instance().setJitVariable(ID, RC);
}

extern "C" void __proteus_enable_host() {
  JitEngineHost &Jit = JitEngineHost::instance();
  Jit.enable();
}

extern "C" __attribute__((noinline)) void
__proteus_take_address(void const *) noexcept {}

extern "C" void __proteus_disable_host() {
  JitEngineHost &Jit = JitEngineHost::instance();
  Jit.disable();
}

// NOLINTEND(readability-identifier-naming)
