#ifndef PROTEUS_DEBUG_H
#define PROTEUS_DEBUG_H

#include <llvm/IR/Value.h>
#include <llvm/Support/raw_ostream.h>

#if PROTEUS_ENABLE_DEBUG
#define PROTEUS_DBG(x) x;
#else
#define PROTEUS_DBG(x)
#endif

namespace proteus {

using namespace llvm;

inline std::string toString(Value &V) {
  std::string Str;
  raw_string_ostream OS{Str};
  V.print(OS);
  return Str;
}

} // namespace proteus

#endif
