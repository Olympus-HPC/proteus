#ifndef PROTEUS_DEBUG_H
#define PROTEUS_DEBUG_H

#include "proteus/impl/Config.h"

#include <llvm/IR/Value.h>
#include <llvm/Support/raw_ostream.h>

#define PROTEUS_DBG(x)                                                         \
  do                                                                           \
    if (Config::get().ProteusDebugOutput) {                                    \
      x;                                                                       \
    }                                                                          \
  while (0);

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
