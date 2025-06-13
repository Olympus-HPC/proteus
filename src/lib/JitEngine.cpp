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
  Config.PROTEUS_SET_LAUNCH_BOUNDS =
      getEnvOrDefaultBool("PROTEUS_SET_LAUNCH_BOUNDS", true);
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
  Logger::logs("proteus") << "PROTEUS_SET_LAUNCH_BOUNDS "
                          << Config.PROTEUS_SET_LAUNCH_BOUNDS << "\n";
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

void JitEngine::optimizeIR(Module &M, StringRef Arch, StringRef OptLevel,
                           unsigned CodegenOptLevel) {
  TIMESCOPE("Optimize IR");
  proteus::optimizeIR(M, Arch, OptLevel, CodegenOptLevel);
}

void JitEngine::runCleanupPassPipeline(Module &M) {
  proteus::runCleanupPassPipeline(M);
}

} // namespace proteus
