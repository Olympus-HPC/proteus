#ifndef PROTEUS_IMPL_MLIR_IR_FUNCTION_H
#define PROTEUS_IMPL_MLIR_IR_FUNCTION_H

#include "proteus/Frontend/IRFunction.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace proteus {

/// MLIR-backend implementation of IRFunction.
/// Carries the mlir::func::FuncOp handle (a lightweight value type).
class MLIRIRFunction : public IRFunction {
public:
  mlir::func::FuncOp F;
  explicit MLIRIRFunction(mlir::func::FuncOp F) : F(F) {}
};

} // namespace proteus

#endif // PROTEUS_IMPL_MLIR_IR_FUNCTION_H
