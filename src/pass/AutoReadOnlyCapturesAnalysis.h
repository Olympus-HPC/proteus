#ifndef PROTEUS_PASS_AUTO_READONLY_CAPTURES_ANALYSIS_H
#define PROTEUS_PASS_AUTO_READONLY_CAPTURES_ANALYSIS_H

#include "proteus/AutoReadOnlyCaptures.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>

#include <cstdint>

namespace llvm {
class Function;
class Module;
} // namespace llvm

namespace proteus {

llvm::SmallVector<AutoReadOnlyCaptureMetadataEntry>
analyzeAutoReadOnlyCaptures(llvm::Function &F);

void emitAutoReadOnlyCapturesMetadata(
    llvm::Function &F,
    llvm::ArrayRef<AutoReadOnlyCaptureMetadataEntry> Captures);

void annotateAutoReadOnlyCaptures(llvm::Module &M);

} // namespace proteus

#endif
