#ifndef PROTEUS_FRONTEND_IRVALUE_H
#define PROTEUS_FRONTEND_IRVALUE_H

namespace proteus {

/// A backend-independent opaque handle to an IR value.
///
/// \c IRValue is an abstract base class.  Backend-specific subclasses hold
/// the native value.  The code builder owns all value objects; the frontend
/// passes raw \c IRValue* pointers around — cheap, trivially copyable, no
/// indirection cost.
///
/// Lifetime: every \c IRValue* is owned by the \c LLVMCodeBuilder (and thus
/// by the \c JitModule) that created it.  Callers must never delete them.
class IRValue {
public:
  virtual ~IRValue() = default;
};

} // namespace proteus

#endif // PROTEUS_FRONTEND_IRVALUE_H
