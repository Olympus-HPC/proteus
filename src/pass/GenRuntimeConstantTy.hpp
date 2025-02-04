//===-- GenRuntimeConstantTy.hpp -- Generate runtime constant type --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_GEN_RUNTIME_CONSTANT_TY_HPP
#define PROTEUS_GEN_RUNTIME_CONSTANT_TY_HPP

#include <llvm/AsmParser/Parser.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>

#include "GenCompilerInterfaceTypes.h"

extern const char GenModule[];

namespace proteus {

using namespace llvm;

// Extract the RuntimeConstantTy using the LLVM IR of a cmake-compiled
// module for the target platform to ensure the type definition between the
// pass and the runtime match.
static Expected<StructType *> getRuntimeConstantTy(LLVMContext &TargetCtx) {
  LLVMContext Context;
  SMDiagnostic Err;
  auto GenM = parseAssemblyString(GenModule, Err, Context);
  if (!GenM) {
    return createStringError(inconvertibleErrorCode(),
                             "Cannot parse generated module: " +
                                 Err.getMessage());
  }

  auto MapTypeToTargetContext =
      [&TargetCtx](Type *Ty,
                   auto &&MapTypeToTargetContext) -> Expected<Type *> {
    if (Ty->isStructTy()) {
      auto *StructTy = cast<StructType>(Ty);

      // Handle unnamed literal struct.
      if (StructTy->isLiteral()) {
        SmallVector<Type *> ElementTypes;
        for (Type *ElemTy : StructTy->elements()) {
          auto ExpectedMappedElemTy =
              MapTypeToTargetContext(ElemTy, MapTypeToTargetContext);
          if (auto E = ExpectedMappedElemTy.takeError())
            return createStringError(
                inconvertibleErrorCode(),
                "Failed to map element type in literal struct: " +
                    toString(std::move(E)));
          ElementTypes.push_back(ExpectedMappedElemTy.get());
        }
        return StructType::get(TargetCtx, ElementTypes, StructTy->isPacked());
      }

      // Handle named struct.
      StructType *ExistingType =
          StructType::getTypeByName(TargetCtx, StructTy->getName());
      if (ExistingType)
        return ExistingType;

      // Recursively populate elements.
      SmallVector<Type *> ElementTypes;
      for (Type *ElemTy : StructTy->elements()) {
        auto ExpectedMappedElemTy =
            MapTypeToTargetContext(ElemTy, MapTypeToTargetContext);
        if (auto E = ExpectedMappedElemTy.takeError()) {
          return createStringError(
              inconvertibleErrorCode(),
              "Failed to map element type in named struct: " +
                  toString(std::move(E)));
        }
        ElementTypes.push_back(ExpectedMappedElemTy.get());
      }

      StructType *NewStruct = StructType::create(
          TargetCtx, ElementTypes, StructTy->getName(), StructTy->isPacked());
      return NewStruct;
    }

    if (Ty->isArrayTy()) {
      ArrayType *ArrayTy = cast<ArrayType>(Ty);
      auto ExpectedElementType = MapTypeToTargetContext(
          ArrayTy->getElementType(), MapTypeToTargetContext);
      if (auto E = ExpectedElementType.takeError()) {
        return createStringError(inconvertibleErrorCode(),
                                 "Failed to map array element type: " +
                                     toString(std::move(E)));
      }

      return ArrayType::get(ExpectedElementType.get(),
                            ArrayTy->getNumElements());
    }

    if (Ty->isPointerTy()) {
      PointerType *PointerTy = cast<PointerType>(Ty);
      return PointerType::get(TargetCtx, PointerTy->getAddressSpace());
    }

    if (Ty->isIntegerTy()) {
      return IntegerType::get(TargetCtx, cast<IntegerType>(Ty)->getBitWidth());
    }

    if (Ty->isFloatingPointTy()) {
      if (Ty->isHalfTy())
        return Type::getHalfTy(TargetCtx);
      if (Ty->isFloatTy())
        return Type::getFloatTy(TargetCtx);
      if (Ty->isDoubleTy())
        return Type::getDoubleTy(TargetCtx);
      if (Ty->isFP128Ty())
        return Type::getFP128Ty(TargetCtx);
      if (Ty->isX86_FP80Ty())
        return Type::getX86_FP80Ty(TargetCtx);
      if (Ty->isPPC_FP128Ty())
        return Type::getPPC_FP128Ty(TargetCtx);
    }

    std::string TyStr;
    raw_string_ostream OS{TyStr};
    Ty->print(OS, true);
    return createStringError(inconvertibleErrorCode(),
                             "Unsupported type: " + TyStr);
  };

  StructType *GenRuntimeConstantTy =
      StructType::getTypeByName(Context, "struct.proteus::RuntimeConstant");
  if (!GenRuntimeConstantTy)
    return createStringError(inconvertibleErrorCode(),
                             "Expected non-null GenRuntimeConstantTy");

  auto ExpectedRuntimeConstantTy =
      MapTypeToTargetContext(GenRuntimeConstantTy, MapTypeToTargetContext);
  if (auto E = ExpectedRuntimeConstantTy.takeError())
    return createStringError(inconvertibleErrorCode(),
                             "Failed to map runtime constant type: " +
                                 toString(std::move(E)));

  return cast<StructType>(ExpectedRuntimeConstantTy.get());
}

} // namespace proteus

#endif
