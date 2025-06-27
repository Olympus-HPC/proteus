//===-- TransformLambdaSpecialization.hpp -- Specialize arguments --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_TRANSFORM_LAMBDA_SPECIALIZATION_HPP
#define PROTEUS_TRANSFORM_LAMBDA_SPECIALIZATION_HPP

#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>

#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/Debug.h"
#include "proteus/Utils.h"
#include <queue>

namespace proteus {

using namespace llvm;

inline Constant *getConstant(LLVMContext &Ctx, Type *ArgType,
                             const RuntimeConstant &RC) {
  if (ArgType->isIntegerTy(1)) {
    return ConstantInt::get(ArgType, RC.Value.BoolVal);
  } else if (ArgType->isIntegerTy(8)) {
    return ConstantInt::get(ArgType, RC.Value.Int8Val);
  } else if (ArgType->isIntegerTy(32)) {
    return ConstantInt::get(ArgType, RC.Value.Int32Val);
  } else if (ArgType->isIntegerTy(64)) {
    return ConstantInt::get(ArgType, RC.Value.Int64Val);
  } else if (ArgType->isFloatTy()) {
    return ConstantFP::get(ArgType, RC.Value.FloatVal);
  } else if (ArgType->isDoubleTy()) {
    return ConstantFP::get(ArgType, RC.Value.DoubleVal);
  } else if (ArgType->isX86_FP80Ty() || ArgType->isPPC_FP128Ty() ||
             ArgType->isFP128Ty()) {
    return ConstantFP::get(ArgType, RC.Value.LongDoubleVal);
  } else if (ArgType->isPointerTy()) {
    auto *IntC = ConstantInt::get(Type::getInt64Ty(Ctx), RC.Value.Int64Val);
    return ConstantExpr::getIntToPtr(IntC, ArgType);
  } else {
    std::string TypeString;
    raw_string_ostream TypeOstream(TypeString);
    ArgType->print(TypeOstream);
    PROTEUS_FATAL_ERROR("JIT Incompatible type in runtime constant: " +
                        TypeOstream.str());
  }
}

//inline Constant *getConstant(LLVMContext &Ctx, const RuntimeConstant &RC) {
//  switch (RC.Value) {
//  case RuntimeConstantTypes::BOOL:
//    return ConstantInt::get(Type::getInt1Ty(Ctx), RC.Value.BoolVal);
//  case RuntimeConstantTypes::INT8:
//    return ConstantInt::get(Type::getInt8Ty(Ctx), RC.Value.Int8Val);
//  case RuntimeConstantTypes::INT32:
//    return ConstantInt::get(Type::getInt32Ty(Ctx), RC.Value.Int32Val);
//  case RuntimeConstantTypes::INT64:
//    return ConstantInt::get(Type::getInt64Ty(Ctx), RC.Value.Int64Val);
//  case RuntimeConstantTypes::FLOAT:
//    return ConstantFP::get(Type::getFloatTy(Ctx), RC.Value.FloatVal);
//  case RuntimeConstantTypes::DOUBLE:
//    return ConstantFP::get(Type::getDoubleTy(Ctx), RC.Value.DoubleVal);
//  case RuntimeConstantTypes::LONG_DOUBLE:
//    return ConstantFP::get(Type::getFP128Ty(Ctx), RC.Value.LongDoubleVal);
//  case RuntimeConstantTypes::PTR:
//    auto *IntC = ConstantInt::get(Type::getInt64Ty(Ctx), RC.Value.Int64Val);
//    return ConstantExpr::getIntToPtr(IntC, ArgType);
//  default:
//    std::string TypeString;
//    raw_string_ostream TypeOstream(TypeString);
//    ArgType->print(TypeOstream);
//    PROTEUS_FATAL_ERROR("JIT Incompatible type in runtime constant: " +
//                        TypeOstream.str());
//  }
//}

class TransformLambdaSpecialization {
public:
  static void transform(Module &M, Function &F,
                        const SmallVector<RuntimeConstant> &RCVec) {
    M.print(llvm::outs(),nullptr);
    auto *LambdaClass = F.getArg(0);
    PROTEUS_DBG(Logger::logs("proteus")
                << "[LambdaSpec] Function: " << F.getName() << " RCVec size "
                << RCVec.size() << "\n");
    PROTEUS_DBG(Logger::logs("proteus")
                << "TransformLambdaSpecialization::transform" << "\n");
    PROTEUS_DBG(Logger::logs("proteus") << "\t args" << "\n");
#if PROTEUS_ENABLE_DEBUG
    for (auto &Arg : RCVec) {
      Logger::logs("proteus")
          << "{" << Arg.Value.Int64Val << ", " << Arg.Slot << " }\n";
    }
#endif
// The below implementation works for lambdas where the entire capture list is a runtime
// constant, and does not have a complex struct type (which precludes all RAJA launch lambdas)
#if 0
    Type* LambdaClassType = nullptr;
    for (User *User : LambdaClass->users()) {
      if (auto *GEP = dyn_cast<GetElementPtrInst>(User)) {
        LambdaClassType = GEP->getSourceElementType();
      }
    }
    if (!LambdaClassType) {
      PROTEUS_FATAL_ERROR("No GEP found for lambda class");
    }
    StructType* LambdaStructType = dyn_cast<StructType>(LambdaClassType);
    if (!LambdaStructType) {
      PROTEUS_FATAL_ERROR("GEP source element type not a struct");
    }

    llvm::outs() << *LambdaStructType << "\n";

    ArrayRef<Type*> NonConstElements = LambdaStructType->elements();
    DenseMap<int, RuntimeConstant> RCMap;
    SmallVector<Constant*> Fields;
    for (const auto& RC: RCVec) {
      RCMap[RC.Slot] = RC;
    }
    for (int32_t Idx = 0; Idx < NonConstElements.size(); ++Idx) {
      Type* ElemType = NonConstElements[Idx];
      if (RCMap.contains(Idx)) {
        Fields.push_back(getConstant(M.getContext(), NonConstElements[Idx], RCMap[Idx]));
      } else {
        PROTEUS_FATAL_ERROR("All fields must be fixed");
      }
    }
    Constant* ConstStruct = ConstantStruct::get(LambdaStructType, Fields);
    GlobalVariable *GV = new GlobalVariable(M,
                                            LambdaStructType,
                                            /*isConstant=*/true,
                                            GlobalValue::InternalLinkage,
                                            ConstStruct,
                                            "my_global_struct",
                                            nullptr,
                                            llvm::GlobalVariable::NotThreadLocal,
                                            0
                                          );
    llvm::outs() << "GV ADDR SPACE " << GV->getAddressSpace() << "\n";


for (User *User : LambdaClass->users()) {
   llvm::outs() << "LAMBDA USR " << *User << "\n";
}
#endif
    DenseSet<Value*> visited;
    DenseMap<Value*, int> ValToSlot;
    std::queue<std::pair<Value*, int>> to_visit;
    for (User *User : LambdaClass->users()) {
      to_visit.push(std::make_pair(User, -1));
    }
    while (!to_visit.empty()) {
      auto* CurUser = to_visit.front().first;
      int slot = to_visit.front().second;
      to_visit.pop();
      if (visited.contains(CurUser))
        continue;
      llvm::outs() << "processing " << *CurUser << "SLOT = "<< slot << "\n";
      visited.insert(CurUser);
      ValToSlot[CurUser] = slot;
      if (auto* LI = dyn_cast<LoadInst>(CurUser); LI ) {
        auto Ty = LI->getType();
        if (Ty->isPointerTy()) {
          for (auto* Usr : CurUser->users())
            to_visit.push(std::make_pair(Usr, slot));
          continue;
        }
        if (slot == -1) {
          slot = 0;
        }
        for (auto &Arg : RCVec) {
          if (Arg.Slot == slot) {
            Constant *C = getConstant(M.getContext(), CurUser->getType(), Arg);
            CurUser->replaceAllUsesWith(C);
            llvm::outs ()
                        << "[LambdaSpec] Replacing " << *CurUser << " with " << *C
                        << "\n";
          }
        }
      } else if (auto *GEP = dyn_cast<GetElementPtrInst>(CurUser); GEP) {
        auto *GEPSlot = GEP->getOperand(GEP->getNumOperands() - 1);
        ConstantInt *CI = dyn_cast<ConstantInt>(GEPSlot);
        int Slot = slot == -1 ? CI->getZExtValue() : slot;

        for (auto* usr : GEP->users()) {
          to_visit.push(std::make_pair(usr, Slot));
          ValToSlot[CurUser] = Slot;
        }

      } else if (auto* Store = dyn_cast<StoreInst>(CurUser); Store ){
        to_visit.push(std::make_pair(Store->getPointerOperand(), slot == -1 ? 0: slot));
        ValToSlot[Store->getPointerOperand()] = slot == -1 ? 0: slot;
        for (auto* PtrUser : Store->getPointerOperand()->users()) {
          // Stores are used to access the first data member in the struct
          to_visit.push(std::make_pair(PtrUser, slot == -1 ? 0: slot));
          ValToSlot[PtrUser] = slot == -1 ? 0: slot;
          llvm::outs() << *PtrUser << "\n";
        }
      } else if (auto* CB = dyn_cast<CallBase>(CurUser); CB) {
        auto* Fn = CB->getCalledFunction();
        for (unsigned i = 0; i < CB->arg_size(); ++i) {
          llvm::Value *argVal = CB->getArgOperand(i);

          if (ValToSlot.contains(argVal)) {
            llvm::outs()<< "ADDING " << *Fn->getArg(i) << "TO LIST\n";
            to_visit.push(std::make_pair(Fn->getArg(i), ValToSlot[argVal]));
          }
        }
      } else {
        for (auto* Usr : CurUser->users()) {
          to_visit.push(std::make_pair(Usr, slot));
        }
      }
    }

      PROTEUS_DBG(Logger::logs("proteus") << "\t users" << "\n");

#if 0
    std::queue<User*> q;
    for (User *Usr : LambdaClass->users()) {
      q.push(Usr);
    }
    while(!q.empty()) {
      auto* User = q.front();
      q.pop();
      llvm::outs() << "LAMBDA USR " << *User << "\n";
      //continue;
      if (auto* LI = dyn_cast<LoadInst>(User); LI ) {
        auto Ty = LI->getType();
        int slot = -1;
        if (Ty->isPointerTy()) {
          slot = 0;
          LI->setOperand(0, GV);
        } else {
          for (auto &Arg : RCVec) {
            if (Arg.Slot == 0) {
              Constant *C = getConstant(M.getContext(), User->getType(), Arg);
              User->replaceAllUsesWith(C);
              llvm::outs ()
                          << "[LambdaSpec] Replacing " << *User << " with " << *C
                          << "\n";
            }
          }
        }
      } else if (auto *GEP = dyn_cast<GetElementPtrInst>(User)) {
        auto *GEPSlot = GEP->getOperand(User->getNumOperands() - 1);
        Type* AnonClassType = GEP->getSourceElementType();
        llvm::outs() << *AnonClassType << "\n";
        ConstantInt *CI = dyn_cast<ConstantInt>(GEPSlot);
        int Slot = CI->getZExtValue();
        llvm::outs() << "KWORD " << *GEP << "\n";
        llvm::outs() << "ADDR SPACE " << GEP->getAddressSpace() << "\n";
        llvm::outs() << "OLD BASE PTR " << *GEP->getPointerOperand() << "\n";
        GEP->setOperand(0, GV);
        //GEP->setSourceElementType()
      } else if (auto *Store = dyn_cast<StoreInst>(User)) {
        llvm::IRBuilder<> Builder(M.getContext());
        llvm::Value *Loaded = Builder.CreateStore(GV, Store->getPointerOperand());
        Store->replaceAllUsesWith(Loaded);

      }
    }
    #endif
  }
};
} // namespace proteus

#endif
