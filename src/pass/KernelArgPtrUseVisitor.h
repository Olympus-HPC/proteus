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
  CallBase *CallerCB;
  Function *Callee;
};

// Given a newly allocated ptr encountered in def-use analysis beginning at a Lambda callsite,
// we need to determine which definition dominates that ptr.
class LambdaPtrUseVisitor : public PtrUseVisitor<LambdaPtrUseVisitor> {
private:
  Module& M;
  DominatorTree DTree;
  Value* PtrBegin;
  Value* SeenUse;
  CallerFrame ArgBeforeCB;
  int64_t Offset;
  LambdaPtrUseAnalysis Result;
  DataLayout DL;
  SmallVector<Value *> WorkList;
  SmallDenseSet<Value *> Seen;
  bool AnalysisSuccess = false;
  bool AnalysisFailed = false;


public:
  // Constructor used whenever a NeedsDefUseAnalysis Value is encountered. We need to track
  // where the calling LambdaArgVisitor came in from, so that our analysis does not
  LambdaPtrUseVisitor(Value *PtrBegin_, Value* SeenUse_, const DataLayout &_DL, Module& M_)
      : PtrBegin(PtrBegin_), SeenUse(SeenUse_), DL(_DL), M(M_), PtrUseVisitor<LambdaPtrUseVisitor>(_DL) {
    WorkList.push_back(PtrBegin_);
    Seen.insert(PtrBegin_);
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

  void visitStoreInst(StoreInst &SI) {
    WorkList.push_back(SI.getPointerOperand());
  }

  void visitLoadInst(LoadInst &LI) {
    if (LI.getType() != PointerType::get(M.getContext(), DL.getProgramAddressSpace())) {
      AnalysisFailed = true;
      AnalysisSuccess = false;
      return;
    }
    for (User* Usr : LI.users()) {
      WorkList.push_back(Usr);
    }
  }

  void visitCallBase(CallBase &CB) {
    auto OldFrame = ArgBeforeCB;
    Function* F = CB.getCalledFunction();
    Value* ArgToTrack = nullptr;
    for (size_t ArgI = 0; ArgI < F->arg_size(); ++ArgI) {
      if (CB.getArgOperand(ArgI) == ArgBeforeCB.)
        ArgToTrack = F->getArg(ArgI);
    }
    WorkList.push_back(ArgToTrack);
  }

  void visitGetElementPtrInst(GetElementPtrInst &GEP) {
    int64_t GEPOffset = 0;
    GetPointerBaseWithConstantOffset(&GEP, GEPOffset, DL);
    Offset += GEPOffset;
    WorkList.push_back(GEP.getPointerOperand());
  }

  // todo: these three methods need to be changed to find a dominating store
  void visitAllocaInst(AllocaInst &Alloca) {
    for (auto *User : Alloca.users())
      if (!Seen.contains(User))
        WorkList.push_back(User);
  }

  void visitBitCastInst(BitCastInst &BC) {
    for (auto *User : BC.users())
      if (!Seen.contains(User))
        WorkList.push_back(User);
  }

  void visitAddrSpaceCastInst(AddrSpaceCastInst &ASC) {
    WorkList.push_back(ASC.getPointerOperand());
    for (auto *User : ASC.users())
      if (!Seen.contains(User))
        WorkList.push_back(User);
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
      AnalysisFailed = true;
      return;
    }

    // Optional safety: only valid if the tracked byte lies within the copied
    // region.
    if (auto *LenC = dyn_cast<ConstantInt>(MT->getLength())) {
      uint64_t Len = LenC->getZExtValue();
      if (Offset < DstOff || uint64_t(Offset - DstOff) >= Len) {
        AnalysisFailed = true;
        return;
      }
    }

    WorkList.push_back(SrcBase->stripPointerCasts());
    Offset = Offset - DstOff + SrcOff;
  }

  void visitIntrinsicInst(IntrinsicInst &) {
    AnalysisFailed = true;
    return;
  }

  void visitInstruction(Instruction &) {
    AnalysisFailed = true;
    return;
  }

  void visitSelectInst(SelectInst &S) {
    int64_t TOff = 0;
    int64_t FOff = 0;

    Value *TBase = GetPointerBaseWithConstantOffset(S.getTrueValue(), TOff, DL);
    Value *FBase =
        GetPointerBaseWithConstantOffset(S.getFalseValue(), FOff, DL);
    TBase = TBase->stripPointerCasts();
    FBase = FBase->stripPointerCasts();

    if (TBase != FBase || TOff != FOff) {
      AnalysisFailed = true; // would need path-sensitive offsets to proceed
      return;
    }

    WorkList.push_back(TBase);
    Offset += TOff; // select == TBase + TOff
  }
};

inline Value* getDominatingUse(
    llvm::Module &M,
    Value *ValueNeedingAnalysis,
    Value *SeenUse) {
  DEBUG(Logger::logs("proteus-pass") << "Beginning analysis " << "\n");

  LambdaPtrUseVisitor Visitor(ValueNeedingAnalysis, SeenUse, M.getDataLayout());
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
      Visitor.visitPtr(*I);
    else
     continue;

    if (!Visitor.success() || Visitor.failed()) {
      DEBUG(Logger::logs("proteus-pass")
            << "  [PTR use analysis] [WARNING]: Dominating use analysis FAILED for " << *ValueNeedingAnalysis << " <-- " << *SeenUse << "\n");
      return nullptr;
    }
    LambdaPtrUseAnalysis Info = Visitor.getAnalysisResult();
    if (!Info.DominatingWrite)
      return nullptr;
    return Info.DominatingWrite;
  }
  return nullptr;
}
} // namespace proteus

#endif
