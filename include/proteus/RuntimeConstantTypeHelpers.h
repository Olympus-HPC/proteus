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
  if (Ty->isArrayTy())
    return RuntimeConstantType::STATIC_ARRAY;
  if (Ty->isVectorTy())
    return RuntimeConstantType::VECTOR;

  std::string TypeString;
  raw_string_ostream TypeOstream(TypeString);
  Ty->print(TypeOstream);
  reportFatalError("Unknown Type " + TypeOstream.str());
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
    reportFatalError("Unsupported");
  case RuntimeConstantType::PTR:
    return PointerType::getUnqual(Ctx);
  default:
    reportFatalError("Unknown RuntimeConstantType " + toString(RCType));
  }
}

inline size_t getSizeInBytes(RuntimeConstantType RCType) {
  switch (RCType) {
  case RuntimeConstantType::BOOL:
    return sizeof(bool);
  case RuntimeConstantType::INT8:
    return sizeof(int8_t);
  case RuntimeConstantType::INT32:
    return sizeof(int32_t);
  case RuntimeConstantType::INT64:
    return sizeof(int64_t);
  case RuntimeConstantType::FLOAT:
    return sizeof(float);
  case RuntimeConstantType::DOUBLE:
    return sizeof(double);
  case RuntimeConstantType::LONG_DOUBLE:
    return sizeof(long double);
  default:
    reportFatalError("Unknown size for RuntimeConstantType " +
                     toString(RCType));
  }
}

template <typename T> T getValue(const RuntimeConstant &RC) {
  switch (RC.Type) {
  case RuntimeConstantType::BOOL:
    return RC.Value.BoolVal;
  case RuntimeConstantType::INT8:
    return RC.Value.Int8Val;
  case RuntimeConstantType::INT32:
    return RC.Value.Int32Val;
  case RuntimeConstantType::INT64:
    return RC.Value.Int64Val;
  case RuntimeConstantType::FLOAT:
    return RC.Value.FloatVal;
  case RuntimeConstantType::DOUBLE:
    return RC.Value.DoubleVal;
  case RuntimeConstantType::LONG_DOUBLE:
    return RC.Value.LongDoubleVal;
  default:
    reportFatalError("Cannot get value for RuntimeConstantType " +
                     toString(RC.Type));
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
  case RuntimeConstantType::STATIC_ARRAY:
    return "STATIC_ARRAY";
  case RuntimeConstantType::VECTOR:
    return "VECTOR";
  case RuntimeConstantType::ARRAY:
    return "ARRAY";
  case RuntimeConstantType::OBJECT:
    return "OBJECT";
  default:
    reportFatalError("Unknown RCType " + std::to_string(RCType));
  }
}

inline bool isScalarRuntimeConstantType(RuntimeConstantType RCType) {
  return (RCType >= RuntimeConstantType::BOOL &&
          RCType <= RuntimeConstantType::PTR);
}

} // namespace proteus

#endif
