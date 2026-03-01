#ifndef PROTEUS_FRONTEND_LLVMTYPEMAP_H
#define PROTEUS_FRONTEND_LLVMTYPEMAP_H

#include "proteus/Frontend/IRType.h"
#include "proteus/Frontend/TypeMap.h"

#include <cstddef>

namespace llvm {
class LLVMContext;
class Type;
} // namespace llvm

namespace proteus {

/// Convert a backend-independent \c IRType descriptor to its corresponding
/// \c llvm::Type*.
///
/// \param T   The abstract type descriptor produced by \c TypeMap<T>::get().
/// \param Ctx The LLVM context to use when constructing the type.
/// \returns   The corresponding \c llvm::Type*, or \c nullptr when \p T
///            represents an element type absent in the Pointer/Array descriptor
///            (i.e. \c IRTypeKind::Void in that position).
llvm::Type *toLLVMType(const IRType &T, llvm::LLVMContext &Ctx);

// ---------------------------------------------------------------------------
// Convenience wrappers
// ---------------------------------------------------------------------------

/// Return the \c llvm::Type* for a C++ type \c T.
/// This is the LLVM-backend counterpart of \c TypeMap<T>::get().
template <typename T>
llvm::Type *getLLVMType(llvm::LLVMContext &Ctx, std::size_t NElem = 0) {
  return toLLVMType(TypeMap<T>::get(NElem), Ctx);
}

/// Return the \c llvm::Type* of the element type for a C++ pointer type \c T,
/// or \c nullptr when \c T is not a pointer type.
/// This is the LLVM-backend counterpart of \c TypeMap<T>::getPointerElemType().
template <typename T>
llvm::Type *getLLVMPointerElemType(llvm::LLVMContext &Ctx) {
  auto Elem = TypeMap<T>::getPointerElemType();
  return Elem ? toLLVMType(*Elem, Ctx) : nullptr;
}

} // namespace proteus

#endif // PROTEUS_FRONTEND_LLVMTYPEMAP_H
