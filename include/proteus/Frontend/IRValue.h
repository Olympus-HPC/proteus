#ifndef PROTEUS_FRONTEND_IRVALUE_H
#define PROTEUS_FRONTEND_IRVALUE_H

namespace proteus {

/// A backend-independent opaque handle to an IR value.
///
/// Analogously to \c IRType, this struct hides the concrete backend type
/// (\c llvm::Value* in the LLVM backend) from backend-agnostic headers.
/// Conversion between \c IRValue and \c llvm::Value* is provided by
/// \c LLVMValueMap.h.
struct IRValue {
  void *Ptr = nullptr;

  IRValue() = default;
  explicit IRValue(void *P) : Ptr(P) {}

  /// Returns true when this handle refers to a valid IR value.
  bool isValid() const { return Ptr != nullptr; }
};

} // namespace proteus

#endif // PROTEUS_FRONTEND_IRVALUE_H
