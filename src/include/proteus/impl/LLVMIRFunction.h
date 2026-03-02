#ifndef PROTEUS_IMPL_LLVM_IR_FUNCTION_H
#define PROTEUS_IMPL_LLVM_IR_FUNCTION_H

#include "proteus/Frontend/IRFunction.h"

namespace llvm {
class Function;
} // namespace llvm

namespace proteus {

/// LLVM-backend implementation of IRFunction.
/// Carries the raw llvm::Function pointer.
class LLVMIRFunction : public IRFunction {
public:
  llvm::Function *F;
  explicit LLVMIRFunction(llvm::Function *F) : F(F) {}
};

} // namespace proteus

#endif // PROTEUS_IMPL_LLVM_IR_FUNCTION_H
