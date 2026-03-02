#ifndef PROTEUS_IMPL_MLIR_IR_VALUE_H
#define PROTEUS_IMPL_MLIR_IR_VALUE_H

#include "proteus/Frontend/IRValue.h"

#include <mlir/IR/Value.h>

namespace proteus {

/// MLIR-backend concrete IRValue.  Owned by MLIRCodeBuilder::Impl via
/// a std::deque<MLIRIRValue>; frontend code only ever holds an IRValue*
/// pointing into that deque.
///
/// Note: mlir::Value is a lightweight value type (an opaque pointer pair),
/// so we store it directly rather than as a pointer.
class MLIRIRValue : public IRValue {
public:
  mlir::Value V;
  explicit MLIRIRValue(mlir::Value V) : V(V) {}
};

} // namespace proteus

#endif // PROTEUS_IMPL_MLIR_IR_VALUE_H
