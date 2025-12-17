#ifndef PROTEUS_COMPILER_INTERFACE_RUNTIME_CONSTANT_INFO_H
#define PROTEUS_COMPILER_INTERFACE_RUNTIME_CONSTANT_INFO_H

#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/Debug.h"
#include "proteus/Error.h"
#include "proteus/Logger.hpp"
#include "proteus/RuntimeConstantTypeHelpers.h"

#include <optional>

namespace proteus {

using namespace llvm;

// This struct holds the information passed from the compiler pass for a runtime
// constant argument to a function.
struct RuntimeConstantArgInfo {
  RuntimeConstantType Type;
  int32_t Pos;

  explicit RuntimeConstantArgInfo(RuntimeConstantType Type, int32_t Pos)
      : Type(Type), Pos(Pos) {}
};

// This struct holds the information from the compiler pass for a runtime
// constant array, that is the number of elements, if a known compile time
// constant, or a runtime constant argument as the number of elements, and the
// element type.
struct RuntimeConstantArrayInfo {
  int32_t NumElts = 0;
  RuntimeConstantType EltType;

  std::optional<RuntimeConstantArgInfo> OptNumEltsRCInfo = std::nullopt;

  explicit RuntimeConstantArrayInfo(int32_t NumElts,
                                    RuntimeConstantType EltType)
      : NumElts(NumElts), EltType(EltType) {}
  explicit RuntimeConstantArrayInfo(RuntimeConstantType EltType,
                                    RuntimeConstantType NumEltsType,
                                    int32_t NumEltsPos)
      : EltType(EltType),
        OptNumEltsRCInfo{RuntimeConstantArgInfo{NumEltsType, NumEltsPos}} {}
};

// This struct holds the information from the compiler pass for a runtime
// constant object assumed trivially copyable, that is the size of the object
// and whether it is passed by value.
struct RuntimeConstantObjectInfo {
  int32_t Size;
  bool PassByValue;

  explicit RuntimeConstantObjectInfo(int32_t Size, bool PassByValue)
      : Size(Size), PassByValue(PassByValue) {}
};

// This struct holds the information from the compiler pass for a runtime
// constant, be it a scalar or an array. If the runtime constant is an array,
// there is an optional variable to store the runtime constant array info.
struct RuntimeConstantInfo {
  RuntimeConstantArgInfo ArgInfo;
  std::optional<RuntimeConstantArrayInfo> OptArrInfo = std::nullopt;
  std::optional<RuntimeConstantObjectInfo> OptObjInfo = std::nullopt;

  explicit RuntimeConstantInfo(RuntimeConstantType Type, int32_t Pos)
      : ArgInfo{Type, Pos} {
    if (Type == RuntimeConstantType::ARRAY)
      reportFatalError("Missing array info");
  }

  explicit RuntimeConstantInfo(RuntimeConstantType Type, int32_t Pos,
                               int32_t NumElts, RuntimeConstantType EltType)
      : ArgInfo{Type, Pos},
        OptArrInfo{RuntimeConstantArrayInfo{NumElts, EltType}} {
    if ((Type != RuntimeConstantType::ARRAY) &&
        (Type != RuntimeConstantType::STATIC_ARRAY) &&
        (Type != RuntimeConstantType::VECTOR))
      reportFatalError("Expected array runtime constant but type is " +
                       toString(Type));
  }

  explicit RuntimeConstantInfo(RuntimeConstantType Type, int32_t Pos,
                               RuntimeConstantType EltType,
                               RuntimeConstantType NumEltsType,
                               int32_t NumEltsPos)
      : ArgInfo{Type, Pos},
        OptArrInfo{RuntimeConstantArrayInfo{EltType, NumEltsType, NumEltsPos}} {
    if (Type != RuntimeConstantType::ARRAY)
      reportFatalError("Expected array runtime constant but type is " +
                       std::to_string(Type));
  }

  explicit RuntimeConstantInfo(RuntimeConstantType Type, int32_t Pos,
                               int32_t Size, bool PassByValue)
      : ArgInfo{Type, Pos},
        OptObjInfo{RuntimeConstantObjectInfo{Size, PassByValue}} {
    if (Type != RuntimeConstantType::OBJECT)
      reportFatalError("Expected object runtime constant but type is " +
                       std::to_string(Type));
  }

  bool operator==(const RuntimeConstantInfo &O) const {
    return ((ArgInfo.Type == O.ArgInfo.Type) && (ArgInfo.Pos == O.ArgInfo.Pos));
  }
  bool operator!=(const RuntimeConstantInfo &O) const { return !(*this == O); }

  // Compare by Pos.
  bool operator<(const RuntimeConstantInfo &O) const {
    return ArgInfo.Pos < O.ArgInfo.Pos;
  }
};

template <typename T> inline T getRuntimeConstantValue(void *Arg) {
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
    reportFatalError("Unsupported type for runtime constant value");
  }
}

inline RuntimeConstant
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
    reportFatalError("Unsupported runtime constant type: " + toString(RC.Type));
  }

  return RC;
}

}; // namespace proteus

#endif
