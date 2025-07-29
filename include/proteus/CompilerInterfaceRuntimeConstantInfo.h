#ifndef PROTEUS_COMPILER_INTERFACE_RUNTIME_CONSTANT_INFO_H
#define PROTEUS_COMPILER_INTERFACE_RUNTIME_CONSTANT_INFO_H

#include <optional>

#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/Error.h"
#include "proteus/RuntimeConstantTypeHelpers.h"

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
      PROTEUS_FATAL_ERROR("Missing array info");
  }

  explicit RuntimeConstantInfo(RuntimeConstantType Type, int32_t Pos,
                               int32_t NumElts, RuntimeConstantType EltType)
      : ArgInfo{Type, Pos},
        OptArrInfo{RuntimeConstantArrayInfo{NumElts, EltType}} {
    if ((Type != RuntimeConstantType::ARRAY) &&
        (Type != RuntimeConstantType::STATIC_ARRAY) &&
        (Type != RuntimeConstantType::VECTOR))
      PROTEUS_FATAL_ERROR("Expected array runtime constant but type is " +
                          toString(Type));
  }

  explicit RuntimeConstantInfo(RuntimeConstantType Type, int32_t Pos,
                               RuntimeConstantType EltType,
                               RuntimeConstantType NumEltsType,
                               int32_t NumEltsPos)
      : ArgInfo{Type, Pos},
        OptArrInfo{RuntimeConstantArrayInfo{EltType, NumEltsType, NumEltsPos}} {
    if (Type != RuntimeConstantType::ARRAY)
      PROTEUS_FATAL_ERROR("Expected array runtime constant but type is " +
                          std::to_string(Type));
  }

  explicit RuntimeConstantInfo(RuntimeConstantType Type, int32_t Pos,
                               int32_t Size, bool PassByValue)
      : ArgInfo{Type, Pos},
        OptObjInfo{RuntimeConstantObjectInfo{Size, PassByValue}} {
    if (Type != RuntimeConstantType::OBJECT)
      PROTEUS_FATAL_ERROR("Expected object runtime constant but type is " +
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

}; // namespace proteus

#endif
