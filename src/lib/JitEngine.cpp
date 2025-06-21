//===-- JitEngine.cpp -- Base JIT Engine implementation --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include <cstdlib>
#include <string>

#include "proteus/Config.hpp"
#include "proteus/CoreLLVM.hpp"
#include "proteus/Hashing.hpp"
#include "proteus/JitEngine.hpp"
#include "proteus/TimeTracing.hpp"
#include "proteus/Utils.h"

namespace proteus {

#if PROTEUS_ENABLE_TIME_TRACING
TimeTracerRAII TimeTracer;
#endif

using namespace llvm;

JitEngine::JitEngine() {
#if PROTEUS_ENABLE_DEBUG
  Logger::logs("proteus") << "PROTEUS_USE_STORED_CACHE "
                          << Config::get().ProteusUseStoredCache << "\n";
  Logger::logs("proteus") << "PROTEUS_SET_LAUNCH_BOUNDS "
                          << Config::get().ProteusSpecializeLaunchBounds
                          << "\n";
  Logger::logs("proteus") << "PROTEUS_SPECIALIZE_ARGS "
                          << Config::get().ProteusSpecializeArgs << "\n";
  Logger::logs("proteus") << "PROTEUS_SPECIALIZE_DIMS "
                          << Config::get().ProteusSpecializeDims << "\n";
  Logger::logs("proteus") << "PROTEUS_CODEGEN"
                          << toString(Config::get().ProteusCodegen) << "\n";
#endif
}

std::string JitEngine::mangleSuffix(HashT &HashValue) {
  return "$jit$" + HashValue.toString() + "$";
}

void JitEngine::optimizeIR(Module &M, StringRef Arch, char OptLevel,
                           unsigned CodegenOptLevel) {
  TIMESCOPE("Optimize IR");
  proteus::optimizeIR(M, Arch, OptLevel, CodegenOptLevel);
}

SmallVector<RuntimeConstant> JitEngine::getRuntimeConstantValues(
    void **Args, ArrayRef<RuntimeConstantInfo *> RCInfoArray) {
  TIMESCOPE(__FUNCTION__);

  SmallVector<RuntimeConstant> RCVec;
  RCVec.reserve(RCInfoArray.size());
  for (const auto *RCInfo : RCInfoArray) {
    auto &RC = RCVec.emplace_back(RCInfo->Type, RCInfo->Pos);
    PROTEUS_DBG(Logger::logs("proteus")
                << "RC Index " << RC.Pos << " Type " << RC.Type << " ");

    switch (RC.Type) {
    case RuntimeConstantType::BOOL:
      RC.Value.BoolVal = *(bool *)Args[RC.Pos];
      PROTEUS_DBG(Logger::logs("proteus")
                  << "Value " << RC.Value.BoolVal << "\n");
      break;
    case RuntimeConstantType::INT8:
      RC.Value.Int8Val = *(int8_t *)Args[RC.Pos];
      PROTEUS_DBG(Logger::logs("proteus")
                  << "Value " << RC.Value.Int8Val << "\n");
      break;
      break;
    case RuntimeConstantType::INT32:
      RC.Value.Int32Val = *(int32_t *)Args[RC.Pos];
      PROTEUS_DBG(Logger::logs("proteus")
                  << "Value " << RC.Value.Int32Val << "\n");
      break;
    case RuntimeConstantType::INT64:
      RC.Value.Int64Val = *(int64_t *)Args[RC.Pos];
      PROTEUS_DBG(Logger::logs("proteus")
                  << "Value " << RC.Value.Int64Val << "\n");
      break;
    case RuntimeConstantType::FLOAT:
      RC.Value.FloatVal = *(float *)Args[RC.Pos];
      PROTEUS_DBG(Logger::logs("proteus")
                  << "Value " << RC.Value.FloatVal << "\n");
      break;
    case RuntimeConstantType::DOUBLE:
      RC.Value.DoubleVal = *(double *)Args[RC.Pos];
      PROTEUS_DBG(Logger::logs("proteus")
                  << "Value " << RC.Value.DoubleVal << "\n");
      break;
    // NOTE: long double on device should correspond to plain double.
    // XXX: CUDA with a long double SILENTLY fails to create a working
    // kernel in AOT compilation, with or without JIT.
    case RuntimeConstantType::LONG_DOUBLE:
      RC.Value.LongDoubleVal = *(long double *)Args[RC.Pos];
      PROTEUS_DBG(Logger::logs("proteus")
                  << "Value " << std::to_string(RC.Value.LongDoubleVal)
                  << "\n");
      break;
    case RuntimeConstantType::PTR:
      RC.Value.PtrVal = (void *)*(intptr_t *)Args[RC.Pos];
      PROTEUS_DBG(Logger::logs("proteus")
                  << "Value " << RC.Value.PtrVal << "\n");
      break;
    default:
      PROTEUS_FATAL_ERROR("JIT Incompatible type in runtime constant: " +
                          std::to_string(RC.Type));
    }
  }

  return RCVec;
}

} // namespace proteus
