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
  Logger::logs("proteus") << "PROTEUS_OPT_PIPELINE"
                          << Config::get().ProteusOptPipeline << "\n";

#endif
}

std::string JitEngine::mangleSuffix(HashT &HashValue) {
  return "$jit$" + HashValue.toString() + "$";
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

inline static RuntimeConstant
dispatchGetRuntimeConstantValue(void **Args,
                                const RuntimeConstantInfo &RCInfo) {
  RuntimeConstant RC{RCInfo.ArgInfo.Type, RCInfo.ArgInfo.Pos};

  void *Arg = Args[RC.Pos];
  switch (RC.Type) {
  case RuntimeConstantType::BOOL:
    RC.Value.BoolVal = getRuntimeConstantValue<bool>(Arg);
    PROTEUS_DBG(Logger::logs("proteus")
                << "Value " << RC.Value.BoolVal << "\n");
    break;
  case RuntimeConstantType::INT8:
    RC.Value.Int8Val = getRuntimeConstantValue<int8_t>(Arg);
    PROTEUS_DBG(Logger::logs("proteus")
                << "Value " << RC.Value.Int8Val << "\n");
    break;
  case RuntimeConstantType::INT32:
    RC.Value.Int32Val = getRuntimeConstantValue<int32_t>(Arg);
    PROTEUS_DBG(Logger::logs("proteus")
                << "Value " << RC.Value.Int32Val << "\n");
    break;
  case RuntimeConstantType::INT64:
    RC.Value.Int64Val = getRuntimeConstantValue<int64_t>(Arg);
    PROTEUS_DBG(Logger::logs("proteus")
                << "Value " << RC.Value.Int64Val << "\n");
    break;
  case RuntimeConstantType::FLOAT:
    RC.Value.FloatVal = getRuntimeConstantValue<float>(Arg);
    PROTEUS_DBG(Logger::logs("proteus")
                << "Value " << RC.Value.FloatVal << "\n");
    break;
  case RuntimeConstantType::DOUBLE:
    RC.Value.DoubleVal = getRuntimeConstantValue<double>(Arg);
    PROTEUS_DBG(Logger::logs("proteus")
                << "Value " << RC.Value.DoubleVal << "\n");
    break;
  case RuntimeConstantType::LONG_DOUBLE:
    // NOTE: long double on device should correspond to plain double.
    // XXX: CUDA with a long double SILENTLY fails to create a working
    // kernel in AOT compilation, with or without JIT.
    RC.Value.LongDoubleVal = getRuntimeConstantValue<long double>(Arg);
    PROTEUS_DBG(Logger::logs("proteus")
                << "Value " << std::to_string(RC.Value.LongDoubleVal) << "\n");
    break;
  case RuntimeConstantType::PTR:
    RC.Value.PtrVal = (void *)getRuntimeConstantValue<intptr_t>(Arg);
    PROTEUS_DBG(Logger::logs("proteus") << "Value " << RC.Value.PtrVal << "\n");
    break;
  case RuntimeConstantType::ARRAY: {
    int32_t NumElts;
    if (RCInfo.OptArrInfo->OptNumEltsRCInfo) {
      int32_t NumEltsPos = RCInfo.OptArrInfo->OptNumEltsRCInfo->Pos;
      RuntimeConstantType NumEltsType =
          RCInfo.OptArrInfo->OptNumEltsRCInfo->Type;

      RuntimeConstantInfo NumEltsRCInfo{NumEltsType, NumEltsPos};
      RuntimeConstant NumEltsRC =
          dispatchGetRuntimeConstantValue(Args, NumEltsRCInfo);

      NumElts = getValue<int32_t>(NumEltsRC);
    } else {
      NumElts = RCInfo.OptArrInfo->NumElts;
    }

    size_t SizeInBytes = NumElts * getSizeInBytes(RCInfo.OptArrInfo->EltType);
    std::shared_ptr<unsigned char[]> Blob{new unsigned char[SizeInBytes]};
    // The interface is a pointer-to-pointer so we need to deref it to copy the
    // data.
    void *Src = (void *)getRuntimeConstantValue<intptr_t>(Arg);
    std::memcpy(Blob.get(), Src, SizeInBytes);

    RC.ArrInfo = ArrayInfo{NumElts, RCInfo.OptArrInfo->EltType, Blob};

    PROTEUS_DBG(Logger::logs("proteus")
                << "Value Blob ptr " << Blob.get() << "\n");
    break;
  }
  case RuntimeConstantType::STATIC_ARRAY: {
    size_t SizeInBytes =
        RCInfo.OptArrInfo->NumElts * getSizeInBytes(RCInfo.OptArrInfo->EltType);
    std::shared_ptr<unsigned char[]> Blob{new unsigned char[SizeInBytes]};
    // Static arrays are passed by value, so it is a pointer directly to the
    // stack.
    std::memcpy(Blob.get(), Arg, SizeInBytes);

    RC.ArrInfo =
        ArrayInfo{RCInfo.OptArrInfo->NumElts, RCInfo.OptArrInfo->EltType, Blob};

    PROTEUS_DBG(Logger::logs("proteus")
                << "Value Blob ptr " << Blob.get() << "\n");
    break;
  }
  case RuntimeConstantType::VECTOR: {
    size_t SizeInBytes =
        RCInfo.OptArrInfo->NumElts * getSizeInBytes(RCInfo.OptArrInfo->EltType);
    std::shared_ptr<unsigned char[]> Blob{new unsigned char[SizeInBytes]};
    // Vectors are passed by value, so it is a pointer directly to the stack.
    std::memcpy(Blob.get(), Arg, SizeInBytes);

    RC.ArrInfo =
        ArrayInfo{RCInfo.OptArrInfo->NumElts, RCInfo.OptArrInfo->EltType, Blob};

    PROTEUS_DBG(Logger::logs("proteus")
                << "Value Blob ptr " << Blob.get() << "\n");
    break;
  }
  case RuntimeConstantType::OBJECT: {
    std::shared_ptr<unsigned char[]> Blob{
        new unsigned char[RCInfo.OptObjInfo->Size]};

    void *Src = (RCInfo.OptObjInfo->PassByValue
                     ? Args[RCInfo.ArgInfo.Pos]
                     : (void *)getRuntimeConstantValue<intptr_t>(
                           Args[RCInfo.ArgInfo.Pos]));
    std::memcpy(Blob.get(), Src, RCInfo.OptObjInfo->Size);

    RC.ObjInfo = ObjectInfo{RCInfo.OptObjInfo->Size,
                            RCInfo.OptObjInfo->PassByValue, Blob};

    PROTEUS_DBG(Logger::logs("proteus") << "Value " << RC.Value.PtrVal << "\n");
    break;
  }
  default:
    PROTEUS_FATAL_ERROR("Unsupported runtime constant type: " +
                        toString(RC.Type));
  }

  return RC;
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
