#include "proteus/Frontend/LLVMTypeMap.h"

#include "proteus/Frontend/IRType.h"

#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Type.h>

#include "proteus/Error.h"

namespace proteus {
using namespace llvm;

/// Return the \c llvm::Type* corresponding to a scalar \c IRTypeKind.
/// Used internally by \c toLLVMType for both direct scalar types and the
/// element types of arrays/pointers.
static llvm::Type *scalarLLVMType(IRTypeKind Kind, LLVMContext &Ctx) {
  switch (Kind) {
  case IRTypeKind::Void:
    return Type::getVoidTy(Ctx);
  case IRTypeKind::Int1:
    return Type::getInt1Ty(Ctx);
  case IRTypeKind::Int16:
    return Type::getInt16Ty(Ctx);
  case IRTypeKind::Int32:
    return Type::getInt32Ty(Ctx);
  case IRTypeKind::Int64:
    return Type::getInt64Ty(Ctx);
  case IRTypeKind::Float:
    return Type::getFloatTy(Ctx);
  case IRTypeKind::Double:
    return Type::getDoubleTy(Ctx);
  case IRTypeKind::Pointer:
  case IRTypeKind::Array:
    reportFatalError("scalarLLVMType: Pointer/Array are not scalar kinds");
  }
  reportFatalError("scalarLLVMType: unknown IRTypeKind");
}

llvm::Type *toLLVMType(const IRType &T, LLVMContext &Ctx) {
  switch (T.Kind) {
  case IRTypeKind::Void:
  case IRTypeKind::Int1:
  case IRTypeKind::Int16:
  case IRTypeKind::Int32:
  case IRTypeKind::Int64:
  case IRTypeKind::Float:
  case IRTypeKind::Double:
    return scalarLLVMType(T.Kind, Ctx);

  case IRTypeKind::Pointer:
    return PointerType::get(Ctx, T.AddrSpace);

  case IRTypeKind::Array:
    return ArrayType::get(scalarLLVMType(T.ElemKind, Ctx), T.NElem);
  }
  reportFatalError("toLLVMType: unknown IRTypeKind");
}

} // namespace proteus
