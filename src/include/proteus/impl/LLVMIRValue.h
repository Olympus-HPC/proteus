#ifndef PROTEUS_IMPL_LLVM_IR_VALUE_H
#define PROTEUS_IMPL_LLVM_IR_VALUE_H

#include "proteus/Frontend/IRValue.h"

#include <llvm/IR/Value.h>

namespace proteus {

/// LLVM-backend concrete \c IRValue.  Owned by \c LLVMCodeBuilder::Impl via a
/// \c std::deque<LLVMIRValue>. Frontend code only ever holds an \c IRValue*
/// pointing into that deque.
class LLVMIRValue : public IRValue {
public:
  llvm::Value *V;
  explicit LLVMIRValue(llvm::Value *V) : V(V) {}
};

} // namespace proteus

#endif // PROTEUS_IMPL_LLVM_IR_VALUE_H
