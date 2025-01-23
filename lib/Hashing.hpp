#ifndef PROTEUS_HASHING_HPP
#define PROTEUS_HASHING_HPP

#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/TimeTracing.hpp"

#include <llvm/ADT/ArrayRef.h>
#include <string>
#if LLVM_VERSION_MAJOR == 18
#include "llvm/ADT/StableHashing.h"
#else
#include "llvm/CodeGen/StableHashing.h"
#endif

namespace proteus {

using namespace llvm;

class HashT {
private:
  stable_hash Value;

public:
  inline HashT(const stable_hash HashValue) { Value = HashValue; }
  inline HashT(const HashT &H) { Value = H.Value; }
  inline HashT(StringRef S) { S.getAsInteger(0, Value); }
  inline const stable_hash getValue() const { return Value; }
  inline std::string toString() const { return std::to_string(Value); }
  inline bool operator==(const HashT &Other) const {
    return Value == Other.Value;
  }
};

static HashT inline hashValue(HashT &H) { return H; }

static HashT inline hashValue(StringRef &S) {
  return stable_hash_combine_string(S);
}

static HashT inline hashValue(const std::string &S) {
  return stable_hash_combine_string(S);
}

template <typename T>
static inline std::enable_if_t<std::is_scalar<T>::value, HashT>
hashValue(const T &V) {
  return stable_hash_combine_string(
      StringRef{reinterpret_cast<const char *>(&V), sizeof(T)});
}

static HashT inline hashArrayRefElement(const RuntimeConstant &RC) {
  return stable_hash_combine_string(
      StringRef{reinterpret_cast<const char *>(&RC.Value), sizeof(RC.Value)});
}

static HashT inline hashValue(const ArrayRef<RuntimeConstant> &Arr) {
  if (Arr.empty())
    return 0;

  HashT HashValue = hashArrayRefElement(Arr[0]);
  for (int I = 1, E = Arr.size(); I < E; ++I)
    HashValue = stable_hash_combine(HashValue.getValue(),
                                    hashArrayRefElement(Arr[I]).getValue());

  return HashValue;
}

static HashT inline hashCombine(HashT A, HashT B) {
  return stable_hash_combine(A.getValue(), B.getValue());
}

template <typename FirstT, typename... RestTs>
static HashT inline hash(FirstT &&First, RestTs &&...Rest) {
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

template <typename T> static HashT inline hash(T &&Data) {
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
