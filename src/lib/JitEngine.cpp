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

#include "proteus/Hashing.hpp"
#include "proteus/JitEngine.hpp"
#include "proteus/TimeTracing.hpp"

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
  Logger::logs("proteus") << "PROTEUS_OPT_PIPELINE"
                          << Config::get().ProteusOptPipeline << "\n";
  Logger::logs("proteus") << "PROTEUS_CACHE_DIR"
                          << Config::get().ProteusCacheDir << "\n";
#endif
}

std::string JitEngine::mangleSuffix(HashT &HashValue) {
  return "$jit$" + HashValue.toString() + "$";
}

SmallVector<RuntimeConstant> JitEngine::getRuntimeConstantValues(
    void **Args, ArrayRef<RuntimeConstantInfo *> RCInfoArray) {
  TIMESCOPE(__FUNCTION__);

  SmallVector<RuntimeConstant> RCVec;
  RCVec.reserve(RCInfoArray.size());
  for (const auto *RCInfo : RCInfoArray) {
    PROTEUS_DBG(Logger::logs("proteus")
                << "RC Index " << RCInfo->ArgInfo.Pos << " Type "
                << toString(RCInfo->ArgInfo.Type) << " ");

    RCVec.emplace_back(dispatchGetRuntimeConstantValue(Args, *RCInfo));
  }

  return RCVec;
}

} // namespace proteus
