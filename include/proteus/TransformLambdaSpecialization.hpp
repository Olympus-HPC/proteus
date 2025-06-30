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
    // Check for all allocated types
    DenseSet<Type*> NestedLambdaTypes;
    std::queue<std::pair<Value*, Type*>> Worklist;
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (auto* Alloc = dyn_cast<AllocaInst>(&I); Alloc) {
          Type* AllocatedType = Alloc->getAllocatedType();
          StringRef Name = AllocatedType->getStructName();
          if (Name.contains(".anon")) {
            NestedLambdaTypes.insert(AllocatedType);
            Worklist.push(std::make_pair(Alloc, AllocatedType));
          }
        }
      }
    }
    // Collect all GEP into nested types
    DenseMap<Value*, int> GEPIntoNestedLambda;
    DenseMap<Value*, int> GEPIntoBaseLambda;
    DenseMap<int, int> NestedToBaseIndex;
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (auto* GEP = dyn_cast<GetElementPtrInst>(&I); GEP) {
          auto *GEPSlot = GEP->getOperand(GEP->getNumOperands() - 1);
          ConstantInt *CI = dyn_cast<ConstantInt>(GEPSlot);
          int slot = CI->getZExtValue();
          if (NestedLambdaTypes.contains(GEP->getSourceElementType())) {
            GEPIntoNestedLambda[GEP] = slot;
          } else if (GEP->getPointerOperand() == LambdaClass) {
            // Can this be broken? Is it more reliable for this check to be by type?
            GEPIntoBaseLambda[GEP] = slot;
          }
        }
      }
    }

    DenseSet<Value*> Launches;
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (auto* Store = dyn_cast<StoreInst>(&I); Store) {
          if (GEPIntoNestedLambda.contains(Store->getPointerOperand()) &&
              GEPIntoBaseLambda.contains(Store->getValueOperand())) {
            NestedToBaseIndex[GEPIntoNestedLambda[Store->getPointerOperand()]] = GEPIntoBaseLambda[Store->getValueOperand()];
          }
      }
    }
    // Identify nested lambda launches
    std::queue<std::pair<Value*, Type*>> FunctionsToReplace;
    DenseSet<Value*> visited;
    while (!Worklist.empty()) {
      auto& [Val, Ty] = Worklist.front();
      Worklist.pop();
      if (visited.contains(Val)) {
        continue;
      }
      visited.insert(Val);
      if (auto* CB = dyn_cast<CallBase>(Val); CB) {
        FunctionsToReplace.push(std::make_pair(CB->getCalledFunction()->getArgOperand(0), Ty));
      }
      for (auto* Usr : Val->users()) {
        Worklist.push(std::make_pair(Usr,Ty));
      }
    }


    DenseSet<Value*> visited;
    DenseMap<Value*, int> ValToSlot;
    std::queue<std::pair<Value*, int>> to_visit;
   // std::queue<std::pair<Value*, Type*>> ReplaceLoadsWorklist;
    for (const auto& Pair : FunctionsToReplace) {
      to_visit.push(std::make_pair(Pair.first, -1));
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
      }  else {
        for (auto* Usr : CurUser->users()) {
          to_visit.push(std::make_pair(Usr, slot));
        }
      }
    }

      PROTEUS_DBG(Logger::logs("proteus") << "\t users" << "\n");

  }
};
} // namespace proteus

#endif
