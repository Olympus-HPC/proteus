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

void JitEngine::getRuntimeConstantValues(void **Args,
                                         const ArrayRef<int32_t> RCIndices,
                                         const ArrayRef<int32_t> RCTypes,
                                         SmallVector<RuntimeConstant> &RCVec) {
  TIMESCOPE(__FUNCTION__);
  for (size_t I = 0; I < RCIndices.size(); ++I) {
    PROTEUS_DBG(Logger::logs("proteus") << "RC Index " << RCIndices[I]
                                        << " Type " << RCTypes[I] << " ");
    RuntimeConstant RC;
    switch (RCTypes[I]) {
    case RuntimeConstantTypes::BOOL:
      RC.Value.BoolVal = *(bool *)Args[RCIndices[I]];
      PROTEUS_DBG(Logger::logs("proteus")
                  << "Value " << RC.Value.BoolVal << "\n");
      break;
    case RuntimeConstantTypes::INT8:
      RC.Value.Int8Val = *(int8_t *)Args[RCIndices[I]];
      PROTEUS_DBG(Logger::logs("proteus")
                  << "Value " << RC.Value.Int8Val << "\n");
      break;
      break;
    case RuntimeConstantTypes::INT32:
      RC.Value.Int32Val = *(int32_t *)Args[RCIndices[I]];
      PROTEUS_DBG(Logger::logs("proteus")
                  << "Value " << RC.Value.Int32Val << "\n");
      break;
    case RuntimeConstantTypes::INT64:
      RC.Value.Int64Val = *(int64_t *)Args[RCIndices[I]];
      PROTEUS_DBG(Logger::logs("proteus")
                  << "Value " << RC.Value.Int64Val << "\n");
      break;
    case RuntimeConstantTypes::FLOAT:
      RC.Value.FloatVal = *(float *)Args[RCIndices[I]];
      PROTEUS_DBG(Logger::logs("proteus")
                  << "Value " << RC.Value.FloatVal << "\n");
      break;
    case RuntimeConstantTypes::DOUBLE:
      RC.Value.DoubleVal = *(double *)Args[RCIndices[I]];
      PROTEUS_DBG(Logger::logs("proteus")
                  << "Value " << RC.Value.DoubleVal << "\n");
      break;
    // NOTE: long double on device should correspond to plain double.
    // XXX: CUDA with a long double SILENTLY fails to create a working
    // kernel in AOT compilation, with or without JIT.
    case RuntimeConstantTypes::LONG_DOUBLE:
      RC.Value.LongDoubleVal = *(long double *)Args[RCIndices[I]];
      PROTEUS_DBG(Logger::logs("proteus")
                  << "Value " << std::to_string(RC.Value.LongDoubleVal)
                  << "\n");
      break;
    case RuntimeConstantTypes::PTR:
      RC.Value.PtrVal = (void *)*(intptr_t *)Args[RCIndices[I]];
      PROTEUS_DBG(Logger::logs("proteus")
                  << "Value " << RC.Value.PtrVal << "\n");
      break;
    default:
      PROTEUS_FATAL_ERROR("JIT Incompatible type in runtime constant: " +
                          std::to_string(RCTypes[I]));
    }

    RCVec.push_back(RC);
  }
}

} // namespace proteus
