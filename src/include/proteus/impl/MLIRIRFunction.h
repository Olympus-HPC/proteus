#ifndef PROTEUS_IMPL_MLIR_IR_FUNCTION_H
#define PROTEUS_IMPL_MLIR_IR_FUNCTION_H

#include "proteus/Frontend/IRFunction.h"

#include <mlir/IR/Operation.h>

namespace proteus {

/// MLIR-backend implementation of IRFunction.
/// Carries a function-like operation handle (func.func or gpu.func)
/// and kernel intent selected at construction time.
class MLIRIRFunction : public IRFunction {
public:
  mlir::Operation *Op;
  bool IsKernel;
  explicit MLIRIRFunction(mlir::Operation *Op, bool IsKernel)
      : Op(Op), IsKernel(IsKernel) {}
};

} // namespace proteus

#endif // PROTEUS_IMPL_MLIR_IR_FUNCTION_H
