#ifndef PROTEUS_FRONTEND_IR_FUNCTION_H
#define PROTEUS_FRONTEND_IR_FUNCTION_H

namespace proteus {

/// Opaque function handle returned by CodeBuilder::addFunction.
/// Backend implementations subclass this to carry the concrete function
/// pointer (e.g. llvm::Function *).
class IRFunction {
public:
  virtual ~IRFunction() = default;
};

} // namespace proteus

#endif // PROTEUS_FRONTEND_IR_FUNCTION_H
