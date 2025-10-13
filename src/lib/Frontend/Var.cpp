#include "proteus/Frontend/Var.hpp"
#include "proteus/Error.h"
#include "proteus/Frontend/Func.hpp"
#include "proteus/Frontend/TypeMap.hpp"

namespace proteus {

Value *convert(IRBuilderBase IRB, Value *V, Type *TargetType) {
  Type *ValType = V->getType();

  if (ValType == TargetType) {
    return V;
  }

  if (ValType->isIntegerTy() && TargetType->isFloatingPointTy()) {
    return IRB.CreateSIToFP(V, TargetType);
  }

  if (ValType->isFloatingPointTy() && TargetType->isIntegerTy()) {
    return IRB.CreateFPToSI(V, TargetType);
  }

  if (ValType->isIntegerTy() && TargetType->isIntegerTy()) {
    if (ValType->getIntegerBitWidth() < TargetType->getIntegerBitWidth())
      // TODO: emit the correct signed variant.
      return IRB.CreateIntCast(V, TargetType, false);

    // Truncate if Val has more bits than the target type.
    return IRB.CreateTrunc(V, TargetType);
  }

  if (ValType->isFloatingPointTy() && TargetType->isFloatingPointTy())
    return IRB.CreateFPExt(V, TargetType);

  PROTEUS_FATAL_ERROR("Unsupported conversion");
}

/// Get the common type following C++ usual arithmetic conversions.
Type *getCommonType(const DataLayout &DL, Type *T1, Type *T2) {
  // Give priority to floating point types.
  if (T1->isFloatingPointTy() && T2->isIntegerTy()) {
    return T1;
  }

  if (T2->isFloatingPointTy() && T1->isIntegerTy())
    return T2;

  // Return the wider integer type.
  if (T1->isIntegerTy() && T2->isIntegerTy()) {
    return ((T1->getIntegerBitWidth() >= T2->getIntegerBitWidth()) ? T1 : T2);
  }

  if (T1->isFloatingPointTy() && T2->isFloatingPointTy()) {
    return ((DL.getTypeSizeInBits(T1) >= DL.getTypeSizeInBits(T2)) ? T1 : T2);
  }

  PROTEUS_FATAL_ERROR("Unsupported conversion types");
}

} // namespace proteus
