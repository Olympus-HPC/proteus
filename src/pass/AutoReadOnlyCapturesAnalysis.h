#ifndef PROTEUS_PASS_AUTO_READONLY_CAPTURES_ANALYSIS_H
#define PROTEUS_PASS_AUTO_READONLY_CAPTURES_ANALYSIS_H

#include "proteus/CompilerInterfaceTypes.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>

#include <cstdint>

namespace llvm {
class Function;
} // namespace llvm

namespace proteus {

struct AutoReadOnlyCaptureMetadataEntry {
  int32_t SlotIndex;
  int32_t ByteOffset;
  RuntimeConstantType RCType;
};

llvm::SmallVector<AutoReadOnlyCaptureMetadataEntry>
analyzeAutoReadOnlyCaptures(llvm::Function &F);

void emitAutoReadOnlyCapturesMetadata(
    llvm::Function &F,
    llvm::ArrayRef<AutoReadOnlyCaptureMetadataEntry> Captures);

} // namespace proteus

#endif
