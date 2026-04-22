#ifndef PROTEUS_HASHING_H
#define PROTEUS_HASHING_H

#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/TimeTracing.h"
#include "proteus/impl/RuntimeConstantTypeHelpers.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#if LLVM_VERSION_MAJOR >= 18
#include <llvm/ADT/StableHashing.h>
#else
#include <llvm/CodeGen/StableHashing.h>
#endif

#include <algorithm>
#include <string>
#include <type_traits>

namespace proteus {

using namespace llvm;

class HashT {
private:
  stable_hash Value;

public:
  inline HashT(const stable_hash HashValue) { Value = HashValue; }
  inline HashT(const StringRef &S) { S.getAsInteger(0, Value); }
  inline stable_hash getValue() const { return Value; }
  inline std::string toString() const { return std::to_string(Value); }
  // Returns a suffix for mangled JIT function names. CUDA uses "$" delimiters
  // because "." generates invalid PTX. HIP/Host use "." for demangle-ability.
  inline std::string toMangledSuffix() const {
#if PROTEUS_ENABLE_CUDA
    return "$jit$" + toString() + "$";
#else
    return ".jit." + toString();
#endif
  }
  inline bool operator==(const HashT &Other) const {
    return Value == Other.Value;
  }

  inline bool operator<(const HashT &Other) const {
    return Value < Other.Value;
  }
};

inline HashT hashValue(const HashT &H) { return H; }

// Function that abstracts interface differences in stable hashing across LLVM.
inline HashT hashValue(const StringRef &S) {
#if LLVM_VERSION_MAJOR >= 20
  ArrayRef<uint8_t> Bytes(reinterpret_cast<const uint8_t *>(S.data()),
                          S.size());
  return xxh3_64bits(Bytes);
#else
  return stable_hash_combine_string(S);
#endif
}

inline HashT hashValue(const std::string &S) { return hashValue(StringRef{S}); }

template <typename T>
inline std::enable_if_t<std::is_scalar<T>::value, HashT> hashValue(const T &V) {
  return hashValue(StringRef{reinterpret_cast<const char *>(&V), sizeof(T)});
}

template <typename T>
inline HashT hashRuntimeConstantArray(const RuntimeConstant &RC) {
  if (RC.ArrInfo.NumElts <= 0)
    reportFatalError("Invalid number of elements in array: " +
                     std::to_string(RC.ArrInfo.NumElts));

  if (!RC.ArrInfo.Blob)
    reportFatalError("Expected non-null Blob");

  return hashValue(
      StringRef{reinterpret_cast<const char *>(RC.ArrInfo.Blob.get()),
                sizeof(T) * RC.ArrInfo.NumElts});
}

inline HashT hashRuntimeConstantObject(const RuntimeConstant &RC) {
  if (RC.ObjInfo.Size <= 0)
    reportFatalError("Invalid object size <= 0");

  if (!RC.ObjInfo.Blob)
    reportFatalError("Expected non-null Blob");

  return hashValue(
      StringRef{reinterpret_cast<const char *>(RC.ObjInfo.Blob.get()),
                static_cast<size_t>(RC.ObjInfo.Size)});
}

inline HashT hashArrayRefElement(const RuntimeConstant &RC) {
  if (RC.Type == RuntimeConstantType::ARRAY ||
      RC.Type == RuntimeConstantType::STATIC_ARRAY ||
      RC.Type == RuntimeConstantType::VECTOR) {
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
      reportFatalError("Unsupported array element type: " +
                       toString(RC.ArrInfo.EltType));
    }
  } else if (RC.Type == RuntimeConstantType::OBJECT) {
    return hashRuntimeConstantObject(RC);
  } else if (isScalarRuntimeConstantType(RC.Type)) {
    return hashValue(
        StringRef{reinterpret_cast<const char *>(&RC.Value), sizeof(RC.Value)});
  }

  reportFatalError("Unsupported type " + toString(RC.Type));
}

inline HashT hashValue(const RuntimeConstant &RC) {
  HashT H = hashValue(static_cast<int32_t>(RC.Type));
  H = stable_hash_combine(H.getValue(), hashValue(RC.Pos).getValue());
  H = stable_hash_combine(H.getValue(), hashValue(RC.Offset).getValue());

  if (RC.Type == RuntimeConstantType::ARRAY ||
      RC.Type == RuntimeConstantType::STATIC_ARRAY ||
      RC.Type == RuntimeConstantType::VECTOR) {
    H = stable_hash_combine(H.getValue(),
                            hashValue(RC.ArrInfo.NumElts).getValue());
    H = stable_hash_combine(H.getValue(),
                            hashValue(static_cast<int32_t>(RC.ArrInfo.EltType))
                                .getValue());
    H = stable_hash_combine(H.getValue(), hashArrayRefElement(RC).getValue());
    return H;
  }

  if (RC.Type == RuntimeConstantType::OBJECT) {
    H = stable_hash_combine(H.getValue(), hashValue(RC.ObjInfo.Size).getValue());
    H = stable_hash_combine(H.getValue(),
                            hashValue(static_cast<int32_t>(RC.ObjInfo.PassByValue))
                                .getValue());
    H = stable_hash_combine(H.getValue(), hashArrayRefElement(RC).getValue());
    return H;
  }

  if (isScalarRuntimeConstantType(RC.Type)) {
    H = stable_hash_combine(H.getValue(), hashArrayRefElement(RC).getValue());
    return H;
  }

  // For NONE/unsupported values, the type/pos/offset hash above is all we keep.
  return H;
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

template <typename Key, typename Value>
inline std::enable_if_t<std::is_integral<Key>::value || std::is_enum<Key>::value,
                        HashT>
hashValue(const DenseMap<Key, Value> &Map) {
  if (Map.empty())
    return 0;

  SmallVector<Key, 32> Keys;
  Keys.reserve(Map.size());
  for (const auto &KV : Map)
    Keys.push_back(KV.first);

  std::sort(Keys.begin(), Keys.end());

  HashT H = hashValue(static_cast<uint64_t>(Map.size()));
  for (Key K : Keys) {
    auto It = Map.find(K);
    if (It == Map.end())
      reportFatalError("Internal error: DenseMap key vanished during hashing");

    HashT PairH = stable_hash_combine(hashValue(K).getValue(),
                                      hashValue(It->second).getValue());
    H = stable_hash_combine(H.getValue(), PairH.getValue());
  }

  return H;
}

template <typename Key, typename Value>
inline HashT hashValue(DenseMap<Key, Value> &Map) {
  return hashValue(static_cast<const DenseMap<Key, Value> &>(Map));
}

inline HashT hashCombine(HashT A, HashT B) {
  return stable_hash_combine(A.getValue(), B.getValue());
}

template <typename FirstT, typename... RestTs>
inline HashT hash(FirstT &&First, RestTs &&...Rest) {
  TIMESCOPE("proteus::hash");
  HashT HashValue = hashValue(First);

  ((HashValue = hashCombine(HashValue, hashValue(Rest))), ...);

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
