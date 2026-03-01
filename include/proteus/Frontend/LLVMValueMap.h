#ifndef PROTEUS_FRONTEND_LLVM_VALUE_MAP_H
#define PROTEUS_FRONTEND_LLVM_VALUE_MAP_H

#include "proteus/Frontend/IRValue.h"

#include <llvm/IR/Value.h>

namespace proteus {

/// Convert a backend-independent \c IRValue handle to its corresponding
/// \c llvm::Value*.
inline llvm::Value *toLLVMValue(IRValue V) {
  return static_cast<llvm::Value *>(V.Ptr);
}

/// Wrap an \c llvm::Value* in a backend-independent \c IRValue handle.
inline IRValue fromLLVMValue(llvm::Value *V) { return IRValue{V}; }

} // namespace proteus

#endif // PROTEUS_FRONTEND_LLVM_VALUE_MAP_H
