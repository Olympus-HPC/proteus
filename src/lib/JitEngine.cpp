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

template <typename T> inline static T getRuntimeConstantValue(void *Arg) {
  if constexpr (std::is_same_v<T, bool>) {
    return *static_cast<bool *>(Arg);
  } else if constexpr (std::is_same_v<T, int8_t>) {
    return *static_cast<int8_t *>(Arg);
  } else if constexpr (std::is_same_v<T, int32_t>) {
    return *static_cast<int32_t *>(Arg);
  } else if constexpr (std::is_same_v<T, int64_t>) {
    return *static_cast<int64_t *>(Arg);
  } else if constexpr (std::is_same_v<T, float>) {
    return *static_cast<float *>(Arg);
  } else if constexpr (std::is_same_v<T, double>) {
    return *static_cast<double *>(Arg);
  } else if constexpr (std::is_same_v<T, long double>) {
    return *static_cast<long double *>(Arg);
  } else if constexpr (std::is_pointer_v<T>) {
    return static_cast<T>(*(intptr_t *)Arg);
  } else {
    PROTEUS_FATAL_ERROR("Unsupported type for runtime constant value");
  }
}

inline static void dispatchGetRuntimeConstantValue(void *Arg,
                                                   RuntimeConstantType RCType,
                                                   RuntimeConstantValue &RV) {
  switch (RCType) {
  case RuntimeConstantType::BOOL:
    RV.BoolVal = getRuntimeConstantValue<bool>(Arg);
    PROTEUS_DBG(Logger::logs("proteus") << "Value " << RV.BoolVal << "\n");
    break;
  case RuntimeConstantType::INT8:
    RV.Int8Val = getRuntimeConstantValue<int8_t>(Arg);
    PROTEUS_DBG(Logger::logs("proteus") << "Value " << RV.Int8Val << "\n");
    break;
  case RuntimeConstantType::INT32:
    RV.Int32Val = getRuntimeConstantValue<int32_t>(Arg);
    PROTEUS_DBG(Logger::logs("proteus") << "Value " << RV.Int32Val << "\n");
    break;
  case RuntimeConstantType::INT64:
    RV.Int64Val = getRuntimeConstantValue<int64_t>(Arg);
    PROTEUS_DBG(Logger::logs("proteus") << "Value " << RV.Int64Val << "\n");
    break;
  case RuntimeConstantType::FLOAT:
    RV.FloatVal = getRuntimeConstantValue<float>(Arg);
    PROTEUS_DBG(Logger::logs("proteus") << "Value " << RV.FloatVal << "\n");
    break;
  case RuntimeConstantType::DOUBLE:
    RV.DoubleVal = getRuntimeConstantValue<double>(Arg);
    PROTEUS_DBG(Logger::logs("proteus") << "Value " << RV.DoubleVal << "\n");
    break;
  case RuntimeConstantType::LONG_DOUBLE:
    // NOTE: long double on device should correspond to plain double.
    // XXX: CUDA with a long double SILENTLY fails to create a working
    // kernel in AOT compilation, with or without JIT.
    RV.LongDoubleVal = getRuntimeConstantValue<long double>(Arg);
    PROTEUS_DBG(Logger::logs("proteus")
                << "Value " << std::to_string(RV.LongDoubleVal) << "\n");
    break;
  case RuntimeConstantType::PTR:
    RV.PtrVal = (void *)getRuntimeConstantValue<intptr_t>(Arg);
    PROTEUS_DBG(Logger::logs("proteus") << "Value " << RV.PtrVal << "\n");
    break;
  case RuntimeConstantType::ARRAY:
    RV.PtrVal = (void *)getRuntimeConstantValue<intptr_t>(Arg);
    break;
  default:
    PROTEUS_FATAL_ERROR("Unsupported runtime constant type: " +
                        std::to_string(RCType));
  }
}

SmallVector<RuntimeConstant> JitEngine::getRuntimeConstantValues(
    void **Args, ArrayRef<RuntimeConstantInfo *> RCInfoArray) {
  TIMESCOPE(__FUNCTION__);

  SmallVector<RuntimeConstant> RCVec;
  RCVec.reserve(RCInfoArray.size());
  for (const auto *RCInfo : RCInfoArray) {
    auto &RC = RCVec.emplace_back(RCInfo->ArgInfo.Type, RCInfo->ArgInfo.Pos);

    PROTEUS_DBG(Logger::logs("proteus")
                << "RC Index " << RC.Pos << " Type " << RC.Type << " ");

    void *Arg = Args[RC.Pos];

    dispatchGetRuntimeConstantValue(Arg, RC.Type, RC.Value);
    // Resolve NumElts for runtime constant arrays.
    if (RCInfo->ArgInfo.Type == RuntimeConstantType::ARRAY) {
      if (RCInfo->OptArrInfo->OptNumEltsRCInfo) {
        int32_t NumEltsPos = RCInfo->OptArrInfo->OptNumEltsRCInfo->Pos;
        RuntimeConstantType NumEltsType =
            RCInfo->OptArrInfo->OptNumEltsRCInfo->Type;

        RuntimeConstantValue RV;
        dispatchGetRuntimeConstantValue(Args[NumEltsPos], NumEltsType, RV);

        RC.ArrInfo = ArrayInfo{static_cast<int32_t>(RV.Int64Val),
                               RCInfo->OptArrInfo->EltType};

      } else {
        RC.ArrInfo =
            ArrayInfo{RCInfo->OptArrInfo->NumElts, RCInfo->OptArrInfo->EltType};
      }
    }
  }

  return RCVec;
}

} // namespace proteus
