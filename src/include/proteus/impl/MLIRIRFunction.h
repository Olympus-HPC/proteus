#ifndef PROTEUS_IMPL_MLIR_IR_FUNCTION_H
#define PROTEUS_IMPL_MLIR_IR_FUNCTION_H

#include "proteus/Frontend/IRFunction.h"
#include "proteus/Frontend/IRType.h"

#include <mlir/IR/Operation.h>

#include <vector>

namespace proteus {

/// MLIR-backend implementation of IRFunction.
/// Carries a function-like operation handle (func.func or gpu.func)
/// and kernel intent selected at construction time.
class MLIRIRFunction : public IRFunction {
public:
  mlir::Operation *Op;
  bool IsKernel;
  IRType RetTy;
  std::vector<IRType> ArgTys;

  explicit MLIRIRFunction(mlir::Operation *Op, bool IsKernel, IRType RetTy,
                          std::vector<IRType> ArgTys)
      : Op(Op), IsKernel(IsKernel), RetTy(RetTy), ArgTys(std::move(ArgTys)) {}
};

} // namespace proteus

#endif // PROTEUS_IMPL_MLIR_IR_FUNCTION_H
