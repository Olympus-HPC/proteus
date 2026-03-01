#ifndef PROTEUS_FRONTEND_IRTYPE_H
#define PROTEUS_FRONTEND_IRTYPE_H

#include <cstddef>

namespace proteus {

/// Enumerates the primitive IR type kinds understood by Proteus.
/// This enumeration is backend-independent; individual backends (e.g. LLVM,
/// MLIR) are responsible for mapping these kinds to their own type
/// representations.
enum class IRTypeKind {
  Void,
  Int1,
  Int16,
  Int32,
  Int64,
  Float,
  Double,
  Pointer,
  Array,
};

/// A backend-independent descriptor of an IR type.
///
/// For scalar kinds (Void, Int1, Int16, Int32, Int64, Float, Double) only \c
/// Kind and \c Signed are relevant.
///
/// For \c Pointer, \c ElemKind carries the pointee type kind, \c Signed
/// reflects the signedness of the pointee integer type (if applicable), and
/// \c AddrSpace is the address space of the pointer itself.
///
/// For \c Array, both \c ElemKind and \c NElem are populated.
struct IRType {
  IRTypeKind Kind = IRTypeKind::Void;

  /// Signedness of the type (meaningful for integer kinds and pointer-to-int).
  bool Signed = false;

  /// Number of array elements; only meaningful when Kind == Array.
  std::size_t NElem = 0;

  /// Element type kind; meaningful when Kind == Pointer or Kind == Array.
  IRTypeKind ElemKind = IRTypeKind::Void;

  /// Address space of the pointer itself; only meaningful when Kind == Pointer.
  unsigned AddrSpace = 0;
};

/// Returns true when \p T is an integer kind (Int1, Int16, Int32, or Int64).
inline bool isIntegerKind(const IRType &T) {
  return T.Kind == IRTypeKind::Int1 || T.Kind == IRTypeKind::Int16 ||
         T.Kind == IRTypeKind::Int32 || T.Kind == IRTypeKind::Int64;
}

/// Returns true when \p T is a floating-point kind (Float or Double).
inline bool isFloatingPointKind(const IRType &T) {
  return T.Kind == IRTypeKind::Float || T.Kind == IRTypeKind::Double;
}

} // namespace proteus

#endif // PROTEUS_FRONTEND_IRTYPE_H
