#ifndef PROTEUS_RUNTIME_CONSTANT_TYPE_HELPERS_H
#define PROTEUS_RUNTIME_CONSTANT_TYPE_HELPERS_H

#include <string>

#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Type.h>

#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/Error.h"

namespace proteus {

using namespace llvm;

RuntimeConstantType convertTypeToRuntimeConstantType(Type *Ty);
Type *convertRuntimeConstantTypeToLLVMType(RuntimeConstantType RCType,
                                           LLVMContext &Ctx);
inline std::string toString(const RuntimeConstantType RCType);

inline RuntimeConstantType convertTypeToRuntimeConstantType(Type *Ty) {
  if (Ty->isIntegerTy(1))
    return RuntimeConstantType::BOOL;
  if (Ty->isIntegerTy(8))
    return RuntimeConstantType::INT8;
  if (Ty->isIntegerTy(32))
    return RuntimeConstantType::INT32;
  if (Ty->isIntegerTy(64))
    return RuntimeConstantType::INT64;
  if (Ty->isFloatTy())
    return RuntimeConstantType::FLOAT;
  if (Ty->isDoubleTy())
    return RuntimeConstantType::DOUBLE;
  if (Ty->isFP128Ty() || Ty->isPPC_FP128Ty() || Ty->isX86_FP80Ty())
    return RuntimeConstantType::LONG_DOUBLE;
  if (Ty->isPointerTy())
    return RuntimeConstantType::PTR;

  std::string TypeString;
  raw_string_ostream TypeOstream(TypeString);
  Ty->print(TypeOstream);
  PROTEUS_FATAL_ERROR("Unknown Type " + TypeOstream.str());
}

inline Type *convertRuntimeConstantTypeToLLVMType(RuntimeConstantType RCType,
                                                  LLVMContext &Ctx) {
  switch (RCType) {
  case RuntimeConstantType::BOOL:
    return Type::getInt1Ty(Ctx);
  case RuntimeConstantType::INT8:
    return Type::getInt8Ty(Ctx);
  case RuntimeConstantType::INT32:
    return Type::getInt32Ty(Ctx);
  case RuntimeConstantType::INT64:
    return Type::getInt64Ty(Ctx);
  case RuntimeConstantType::FLOAT:
    return Type::getFloatTy(Ctx);
  case RuntimeConstantType::DOUBLE:
    return Type::getDoubleTy(Ctx);
  case RuntimeConstantType::LONG_DOUBLE:
    PROTEUS_FATAL_ERROR("Unsupported");
  case RuntimeConstantType::PTR:
    return PointerType::getUnqual(Ctx);
  default:
    PROTEUS_FATAL_ERROR("Unknown RuntimeConstantType " + toString(RCType));
  }
}

inline std::string toString(const RuntimeConstantType RCType) {
  switch (RCType) {
  case RuntimeConstantType::BOOL:
    return "BOOL";
  case RuntimeConstantType::INT8:
    return "INT8";
  case RuntimeConstantType::INT32:
    return "INT32";
  case RuntimeConstantType::INT64:
    return "INT64";
  case RuntimeConstantType::FLOAT:
    return "FLOAT";
  case RuntimeConstantType::DOUBLE:
    return "DOUBLE";
  case RuntimeConstantType::LONG_DOUBLE:
    return "LONG_DOUBLE";
  case RuntimeConstantType::PTR:
    return "PTR";
  case RuntimeConstantType::ARRAY:
    return "ARRAY";
  default:
    PROTEUS_FATAL_ERROR("Unknown RCType " + std::to_string(RCType));
  }
}

} // namespace proteus

#endif
