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
  Config.PROTEUS_USE_STORED_CACHE =
      getEnvOrDefaultBool("PROTEUS_USE_STORED_CACHE", true);
  Config.PROTEUS_SPECIALIZE_LAUNCH_BOUNDS =
      getEnvOrDefaultBool("PROTEUS_SPECIALIZE_LAUNCH_BOUNDS", true);
  Config.PROTEUS_USE_POLLY =
      getEnvOrDefaultBool("PROTEUS_USE_POLLY", true);
  Config.PROTEUS_SPECIALIZE_ARGS =
      getEnvOrDefaultBool("PROTEUS_SPECIALIZE_ARGS", true);
  Config.PROTEUS_SPECIALIZE_DIMS =
      getEnvOrDefaultBool("PROTEUS_SPECIALIZE_DIMS", true);
  Config.PROTEUS_USE_HIP_RTC_CODEGEN =
      getEnvOrDefaultBool("PROTEUS_USE_HIP_RTC_CODEGEN", true);
  Config.PROTEUS_DISABLE = getEnvOrDefaultBool("PROTEUS_DISABLE", false);
  Config.PROTEUS_DUMP_LLVM_IR =
      getEnvOrDefaultBool("PROTEUS_DUMP_LLVM_IR", false);
  Config.PROTEUS_RELINK_GLOBALS_BY_COPY =
      getEnvOrDefaultBool("PROTEUS_RELINK_GLOBALS_BY_COPY", false);
  Config.PROTEUS_ASYNC_COMPILATION =
      getEnvOrDefaultBool("PROTEUS_ASYNC_COMPILATION", false);
  Config.PROTEUS_ASYNC_TEST_BLOCKING =
      getEnvOrDefaultBool("PROTEUS_ASYNC_TEST_BLOCKING", false);
  Config.PROTEUS_ASYNC_THREADS = getEnvOrDefaultInt("PROTEUS_ASYNC_THREADS", 1);
  Config.PROTEUS_USE_LIGHTWEIGHT_KERNEL_CLONE =
      getEnvOrDefaultBool("PROTEUS_USE_LIGHTWEIGHT_KERNEL_CLONE", true);

#if PROTEUS_ENABLE_DEBUG
  Logger::logs("proteus") << "PROTEUS_USE_STORED_CACHE "
                          << Config.PROTEUS_USE_STORED_CACHE << "\n";
  Logger::logs("proteus") << "PROTEUS_SPECIALIZE_LAUNCH_BOUNDS "
                          << Config.PROTEUS_SPECIALIZE_LAUNCH_BOUNDS << "\n";
  Logger::logs("proteus") << "PROTEUS_SPECIALIZE_ARGS "
                          << Config.PROTEUS_SPECIALIZE_ARGS << "\n";
  Logger::logs("proteus") << "PROTEUS_SPECIALIZE_DIMS "
                          << Config.PROTEUS_SPECIALIZE_DIMS << "\n";
  Logger::logs("proteus") << "PROTEUS_USE_HIP_RTC_CODEGEN "
                          << Config.PROTEUS_USE_HIP_RTC_CODEGEN << "\n";
#endif
}

std::string JitEngine::mangleSuffix(HashT &HashValue) {
  return "$jit$" + HashValue.toString() + "$";
}

void JitEngine::optimizeIR(Module &M, StringRef Arch, char OptLevel,
                           unsigned CodegenOptLevel) {
  TIMESCOPE("Optimize IR");
  proteus::optimizeIR(M, Arch, OptLevel, CodegenOptLevel, Config.PROTEUS_USE_POLLY);
}

void JitEngine::runCleanupPassPipeline(Module &M) {
  proteus::runCleanupPassPipeline(M);
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
