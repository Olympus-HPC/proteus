#ifndef PROTEUS_KERNELARGPTRDEFVISITOR_H
#define PROTEUS_KERNELARGPTRDEFVISITOR_H

#include "Helpers.h"
#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/impl/Logger.h"
#include "proteus/impl/RuntimeConstantTypeHelpers.h"
#include <alloca.h>
#include <cstdint>
#include <llvm/Analysis/PtrUseVisitor.h>
#include <llvm/Analysis/ValueTracking.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/Hashing.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>

#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Type.h>
#include <memory>
#include <optional>

namespace proteus {
using namespace llvm;

// Any instruction creating a new ptr needs use analysis
bool needsDefUseAnalysis(Value* Val) {
  return isa<AddrSpaceCastInst>(Val) || isa<AllocaInst>(Val) || isa<BitCastInst>(Val) || isa<IntToPtrInst>(Val);
}

struct LambdaPtrUseAnalysis {
  Value *DominatingWrite = nullptr;
  int64_t Offset = 0;
  // Sometimes instructions like ptrtoint --> inttoptr change the layout of
  // the kernel args.
  std::optional<RuntimeConstantType> ChangedRCLayout = std::nullopt;
};

struct CallerFrame {
  Value* ValEnteringCallBase;
  CallBase *CallerCB;
  Function *Callee;
};

// Given a newly allocated ptr encountered in def-use analysis beginning at a Lambda callsite,
// we need to determine which definition dominates that ptr.
class LambdaInstUseVisitor : public InstVisitor<LambdaInstUseVisitor> {
private:
  DominatorTree DTree;
  Value* PtrBegin;
  Value* SeenUse;
  CallerFrame ArgBeforeCB;
  int64_t Offset = 0;
  LambdaPtrUseAnalysis Result;
  DataLayout DL;
  SmallVector<Value *> WorkList;
  SmallDenseSet<Value *> Seen;
  bool AnalysisSuccess = false;
  bool AnalysisFailed = false;


public:
  // Constructor used whenever a NeedsDefUseAnalysis Value is encountered. We need to track
  // where the calling LambdaArgVisitor came in from, so that our analysis does not
  LambdaInstUseVisitor(Value *PtrBegin_, Value* SeenUse_, const DataLayout &_DL)
      : PtrBegin(PtrBegin_), SeenUse(SeenUse_), DL(_DL) {
    WorkList.push_back(PtrBegin_);
    Seen.insert(SeenUse_);
  }
  auto back() { return WorkList.back(); }
  void popBack() { WorkList.pop_back(); }
  bool seen(Value *Val) { return Seen.contains(Val); }
  void markAsSeen(Value *Val) { Seen.insert(Val); }
  bool empty() { return WorkList.empty(); }
  bool success() { return AnalysisSuccess; }
  bool failed() { return AnalysisFailed; }

  auto getAnalysisResult() { return Result; }

  // Keep track of Function frame
  void pushBack(Value* NextVal, Value* CurVal) {
    WorkList.push_back(NextVal);
    if (auto* CB = dyn_cast<CallBase>(NextVal)) {
      // todo: do we need current caller CB
      ArgBeforeCB = CallerFrame{.ValEnteringCallBase = CurVal,
                                .CallerCB = CB,
                                .Callee = CB->getParent()->getParent()};
    }
  }

  void visitStoreInst(StoreInst &SI) {
    AnalysisFailed = false;
    AnalysisSuccess = true;
    Result = {.DominatingWrite = SI.getValueOperand(), .Offset = 0, .ChangedRCLayout = std::nullopt};
    // if (SI.getType()->isPointerTy())
    // pushBack(SI.getPointerOperand(), &SI);
  }

  void visitLoadInst(LoadInst &LI) {
    if (!LI.getType()->isPointerTy()) {
      AnalysisFailed = true;
      AnalysisSuccess = false;
      return;
    }
    for (User* Usr : LI.users()) {
      pushBack(Usr, &LI);
    }
  }

  void visitCallBase(CallBase &CB) {
    auto OldFrame = ArgBeforeCB;
    DEBUG(Logger::logs("proteus-pass") << "    OLD VAL " << *OldFrame.ValEnteringCallBase<< "\n");
    Function* F = CB.getCalledFunction();
    Value* ArgToTrack = nullptr;
    for (size_t ArgI = 0; ArgI < F->arg_size(); ++ArgI) {\
      DEBUG(Logger::logs("proteus-pass") << "    ARG " << ArgI <<" VAL " << *CB.getArgOperand(ArgI) << "\n");
      if (CB.getArgOperand(ArgI) == ArgBeforeCB.ValEnteringCallBase)
        ArgToTrack = F->getArg(ArgI);
    }
    DEBUG(Logger::logs("proteus-pass") << "    Beginning analysis within " << *CB.getCalledFunction() << "\n");

    for (User* Usr : ArgToTrack->users())
      pushBack(Usr, ArgToTrack);
  }

  void visitGetElementPtrInst(GetElementPtrInst &GEP) {
    int64_t GEPOffset = 0;
    GetPointerBaseWithConstantOffset(&GEP, GEPOffset, DL);
    Offset += GEPOffset;
    pushBack(GEP.getPointerOperand(), &GEP);
  }

  // todo: these three methods need to be changed to find a dominating store
  void visitAllocaInst(AllocaInst &Alloca) {
    for (auto *User : Alloca.users())
      if (!Seen.contains(User))
        pushBack(User, &Alloca);
  }

  void visitBitCastInst(BitCastInst &BC) {
    for (auto *User : BC.users())
      if (!Seen.contains(User))
        pushBack(User, &BC);
  }

  void visitAddrSpaceCastInst(AddrSpaceCastInst &ASC) {
    // todo: I don't think the below is necessary, useful, or correct
    // WorkList.push_back(ASC.getPointerOperand());
    DEBUG(Logger::logs("proteus-pass") << ASC << "\n");
    for (auto *User : ASC.users())
      if (!Seen.contains(User)) {
        DEBUG(Logger::logs("proteus-pass") << "  Next up : " <<  *User<< "\n");
        pushBack(User, &ASC);//&ASC);
      }

  }

  void visitMemIntrinsic(MemIntrinsic &I) {
    auto *MT = dyn_cast<MemTransferInst>(&I); // memcpy/memmove
    if (!MT) {
      // memset doesn't preserve any src->dst relationship we can use
      AnalysisFailed = true;
      return;
    }

    int64_t DstOff = 0, SrcOff = 0;
    Value *DstBase =
        GetPointerBaseWithConstantOffset(MT->getRawDest(), DstOff, DL);
    Value *SrcBase =
        GetPointerBaseWithConstantOffset(MT->getRawSource(), SrcOff, DL);
    if (!DstBase || !SrcBase) {
      DEBUG(Logger::logs("proteus-pass") << "  [PTR use analysis]: Failure due to nullptr dst/src " << "\n");
      AnalysisFailed = true;
      return;
    }

    // Optional safety: only valid if the tracked byte lies within the copied
    // region.
    // if (auto *LenC = dyn_cast<ConstantInt>(MT->getLength())) {
    //   uint64_t Len = LenC->getZExtValue();
    //   if (Offset < DstOff || uint64_t(Offset - DstOff) >= Len) {
    //     DEBUG(Logger::logs("proteus-pass") << "  [PTR use analysis]: Failure due to bytesize" << "\n");
    //     // AnalysisFailed = true;
    //     // return;
    //   }
    // }
    DEBUG(Logger::logs("proteus-pass") << "  [PTR use analysis]: Completed instrinsic analysis " << "\n");
    Offset = Offset - DstOff + SrcOff;
    AnalysisSuccess = true;
    AnalysisFailed = false;
    Result = {.DominatingWrite = SrcBase,
              .Offset = Offset,
              .ChangedRCLayout = std::nullopt
              };
  }

  // void visitIntrinsicInst(IntrinsicInst &) {
  //   AnalysisFailed = true;
  //   return;
  // }

  // void visitInstruction(Instruction &) {
  //   AnalysisFailed = true;
  //   return;
  // }

};

inline std::optional<LambdaPtrUseAnalysis> getDominatingUse(
    const DataLayout& DL,
    Value *ValueNeedingAnalysis,
    Value *SeenUse) {
  DEBUG(Logger::logs("proteus-pass") << "Beginning PtrUse analysis " << "\n");

  LambdaInstUseVisitor Visitor(ValueNeedingAnalysis, SeenUse, DL);
  // Analysis loop
  while (!Visitor.empty() && !Visitor.success() && !Visitor.failed()) {
    auto *V = Visitor.back();
    Visitor.popBack();
    // Prevent loops/infinite recursion
    if (Visitor.seen(V))
      continue;
    Visitor.markAsSeen(V);
    DEBUG(Logger::logs("proteus-pass") << "  [PTR use analysis]: Visiting ptr use " << *V << "\n");
    // Analyze the instruction
    if (auto *I = dyn_cast<Instruction>(V))
      Visitor.visit(*I);
    else
     continue;
  }
  if (!Visitor.success() || Visitor.failed()) {
    DEBUG(Logger::logs("proteus-pass")
          << "  [PTR use analysis] [WARNING]: Dominating use analysis FAILED for " << *ValueNeedingAnalysis << " <-- " << *SeenUse << "\n");
    return std::nullopt;
  }
  LambdaPtrUseAnalysis Info = Visitor.getAnalysisResult();
  if (!Info.DominatingWrite)
    return std::nullopt;
  DEBUG(Logger::logs("proteus-pass") << "  [PTR USE ANALYSIS]: Computed offset " << Info.Offset << "\n");
  return Info;

  return std::nullopt;
}
} // namespace proteus

#endif
