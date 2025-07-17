#ifndef PROTEUS_HASHING_HPP
#define PROTEUS_HASHING_HPP

#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/RuntimeConstantTypeHelpers.h"
#include "proteus/TimeTracing.hpp"

#include <llvm/ADT/ArrayRef.h>
#include <string>
#if LLVM_VERSION_MAJOR >= 18
#include <llvm/ADT/StableHashing.h>
#else
#include <llvm/CodeGen/StableHashing.h>
#endif

namespace proteus {

using namespace llvm;

class HashT {
private:
  stable_hash Value;

public:
  inline HashT(const stable_hash HashValue) { Value = HashValue; }
  inline HashT(StringRef S) { S.getAsInteger(0, Value); }
  inline stable_hash getValue() const { return Value; }
  inline std::string toString() const { return std::to_string(Value); }
  inline bool operator==(const HashT &Other) const {
    return Value == Other.Value;
  }

  inline bool operator<(const HashT &Other) const {
    return Value < Other.Value;
  }
};

inline HashT hashValue(HashT &H) { return H; }

inline HashT hashValue(StringRef &S) { return stable_hash_combine_string(S); }

inline HashT hashValue(const std::string &S) {
  return stable_hash_combine_string(S);
}

template <typename T>
inline std::enable_if_t<std::is_scalar<T>::value, HashT> hashValue(const T &V) {
  return stable_hash_combine_string(
      StringRef{reinterpret_cast<const char *>(&V), sizeof(T)});
}

template <typename T>
inline HashT hashRuntimeConstantArray(const RuntimeConstant &RC) {
  if (RC.ArrInfo.NumElts <= 0)
    PROTEUS_FATAL_ERROR("Invalid number of elements in array: " +
                        std::to_string(RC.ArrInfo.NumElts));

  return stable_hash_combine_string(
      StringRef{reinterpret_cast<const char *>(RC.Value.PtrVal),
                sizeof(T) * RC.ArrInfo.NumElts});
}

inline HashT hashArrayRefElement(const RuntimeConstant &RC) {
  if (RC.Type == RuntimeConstantType::ARRAY) {
    switch (RC.ArrInfo.EltType) {
    case RuntimeConstantType::BOOL:
      return hashRuntimeConstantArray<bool>(RC);
    case RuntimeConstantType::INT8:
      return hashRuntimeConstantArray<int8_t>(RC);
    case RuntimeConstantType::INT32:
      return hashRuntimeConstantArray<int32_t>(RC);
    case RuntimeConstantType::INT64:
      return hashRuntimeConstantArray<int64_t>(RC);
    case RuntimeConstantType::FLOAT:
      return hashRuntimeConstantArray<float>(RC);
    case RuntimeConstantType::DOUBLE:
      return hashRuntimeConstantArray<double>(RC);
    default:
      PROTEUS_FATAL_ERROR("Unsupported array element type: " +
                          toString(RC.ArrInfo.EltType));
    }
  }

  return stable_hash_combine_string(
      StringRef{reinterpret_cast<const char *>(&RC.Value), sizeof(RC.Value)});
}

inline HashT hashValue(ArrayRef<RuntimeConstant> Arr) {
  if (Arr.empty())
    return 0;

  HashT HashValue = hashArrayRefElement(Arr[0]);
  for (int I = 1, E = Arr.size(); I < E; ++I)
    HashValue = stable_hash_combine(HashValue.getValue(),
                                    hashArrayRefElement(Arr[I]).getValue());

  return HashValue;
}

inline HashT hashCombine(HashT A, HashT B) {
  return stable_hash_combine(A.getValue(), B.getValue());
}

template <typename FirstT, typename... RestTs>
inline HashT hash(FirstT &&First, RestTs &&...Rest) {
  TIMESCOPE(__FUNCTION__);
  HashT HashValue = hashValue(First);

  (
      [&HashValue, &Rest]() {
        HashValue = stable_hash_combine(HashValue.getValue(),
                                        hashValue(Rest).getValue());
      }(),
      ...);

  return HashValue;
}

template <typename T> inline HashT hash(T &&Data) {
  HashT HashValue = hashValue(Data);
  return HashValue;
}

} // namespace proteus

namespace std {
template <> struct hash<proteus::HashT> {
  std::size_t operator()(const proteus::HashT &Key) const {
    return Key.getValue();
  }
};

} // namespace std
#endif
