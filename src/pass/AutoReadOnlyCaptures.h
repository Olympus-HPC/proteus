#ifndef PROTEUS_PASS_AUTO_READONLY_CAPTURES_H
#define PROTEUS_PASS_AUTO_READONLY_CAPTURES_H

#include "proteus/CompilerInterfaceTypes.h"

#include <llvm/ADT/SmallVector.h>

#include <cstdint>

namespace llvm {
class Function;
} // namespace llvm

namespace proteus {

struct AutoCapture {
  uint32_t Pos;
  uint32_t Offset;
  RuntimeConstantType Type;
};

// Top-level scalar captures of a lambda call operator that are only read, never
// written or escaped, sorted by Pos.
llvm::SmallVector<AutoCapture, 4>
analyzeAutoReadOnlyCaptures(llvm::Function &LambdaOp);

} // namespace proteus

#endif
