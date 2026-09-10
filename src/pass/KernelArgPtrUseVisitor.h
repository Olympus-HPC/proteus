#ifndef PROTEUS_KERNELARGPTRDEFVISITOR_H
#define PROTEUS_KERNELARGPTRDEFVISITOR_H

#include "Helpers.h"
#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/impl/Logger.h"
#include "proteus/impl/RuntimeConstantTypeHelpers.h"
#include <llvm/Analysis/PtrUseVisitor.h>
#include <llvm/Analysis/ValueTracking.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/Hashing.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/AssumptionCache.h>
#include <llvm/Analysis/BasicAliasAnalysis.h>
#include <llvm/Analysis/MemoryLocation.h>
#include <llvm/Analysis/MemorySSA.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
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
#include <llvm/IR/Value.h>
#include <llvm/TargetParser/Triple.h>
#include <memory>
#include <optional>

namespace proteus {
using namespace llvm;

// Any instruction creating a new ptr needs use analysis
bool needsDefUseAnalysis(Value *Val) {
  return isa<AddrSpaceCastInst>(Val) || isa<AllocaInst>(Val) ||
         isa<BitCastInst>(Val) || isa<IntToPtrInst>(Val);
}

inline bool offsetCoveredByRange(int64_t TargetOffset, int64_t RangeOffset,
                                 uint64_t RangeSize) {
  DEBUG(Logger::logs("proteus-pass")
        << "    [PTR use analysis]: Target Offset = " << TargetOffset << "\n");
  DEBUG(Logger::logs("proteus-pass")
        << "    [PTR use analysis]: Range Offset = " << RangeOffset << "\n");
  DEBUG(Logger::logs("proteus-pass")
        << "    [PTR use analysis]: Range Size = " << RangeSize << "\n");
  return TargetOffset >= RangeOffset &&
         static_cast<uint64_t>(TargetOffset - RangeOffset) < RangeSize;
}

inline std::optional<uint64_t> getTypeStoreSize(const DataLayout &DL,
                                                Type *Ty) {
  if (!Ty || !Ty->isSized())
    return std::nullopt;
  return static_cast<uint64_t>(DL.getTypeStoreSize(Ty));
}

struct LambdaPtrUseAnalysis {
  Value *DominatingWrite = nullptr;
  int64_t Offset = 0;
  // Sometimes instructions like ptrtoint --> inttoptr change the layout of
  // the kernel args.
  std::optional<RuntimeConstantType> ChangedRCLayout = std::nullopt;
};

struct CallerFrame {
  Value *ValEnteringCallBase;
  CallBase *CallerCB;
  Function *Callee;
};

// We track use-edges in our analysis.
struct UseEdge {
  Value *CurVal;
  Value *LastVal;
};

inline std::optional<MemoryLocation>
getTrackedPointerLocation(const DataLayout &DL, Value *Ptr) {
  if (!Ptr || !Ptr->getType()->isPointerTy())
    return std::nullopt;

  Type *PointeeTy = nullptr;
  if (auto *AI = dyn_cast<AllocaInst>(Ptr))
    PointeeTy = AI->getAllocatedType();

  if (!PointeeTy || !PointeeTy->isSized())
    return MemoryLocation::getBeforeOrAfter(Ptr);

  return MemoryLocation(Ptr,
                        LocationSize::precise(DL.getTypeStoreSize(PointeeTy)));
}

// Given a newly allocated ptr encountered in def-use analysis beginning at a
// Lambda callsite, we need to determine which definition dominates that ptr.
class LambdaInstUseVisitor : public InstVisitor<LambdaInstUseVisitor> {
private:
  DominatorTree DTree;
  CallerFrame ArgBeforeCB;
  int64_t Offset = 0;
  // The ValueOffsetMap contains the "live range" of the ptr we're analyzing.
  // For example, let's say that LambdaInstUseVisitor is handed ptr %0 = alloca
  // ptr, and LambdaArgVisitor has already identified that the closure starts at
  // byte
  // 8.  In this case, ValueOffsetMap[%0] = 8.  If we encounter a store like
  // store ptr %2, ptr%0, align 8, we don't care, because its written outside
  // of the range of the closure.
  DenseMap<Value *, int> ValueOffsetMap;
  Value *TrackedBase = nullptr;
  LambdaPtrUseAnalysis Result;
  DataLayout DL;
  SmallVector<UseEdge> WorkList;
  // The visitor pattern is always setting LastUse to the back of the
  // edge at the front of the worklist (the def that brought us to the
  // current use).
  Value *Def = nullptr;
  SmallDenseSet<Value *> Seen;
  bool AnalysisSuccess = false;
  bool AnalysisFailed = false;

public:
  // Constructor used whenever a NeedsDefUseAnalysis Value is encountered. We
  // need to track where the calling LambdaArgVisitor came in from, so that our
  // analysis does not
  LambdaInstUseVisitor(Value *PtrBegin, Value *SeenUse, const DataLayout &Dl,
                       int64_t TargetOff)
      : TrackedBase(PtrBegin), DL(Dl) {
    WorkList.push_back({PtrBegin, nullptr});
    Seen.insert(SeenUse);
    ValueOffsetMap[PtrBegin] = TargetOff;
  }
  auto back() { return WorkList.back(); }
  auto popBack() {
    auto Result = WorkList.back();
    Def = Result.LastVal;
    WorkList.pop_back();
    return Result;
  }
  auto getLastDef() { return Def; }
  bool seen(Value *Val) { return Seen.contains(Val); }
  void markAsSeen(Value *Val) { Seen.insert(Val); }
  bool empty() { return WorkList.empty(); }
  bool success() { return AnalysisSuccess; }
  bool failed() { return AnalysisFailed; }

  auto getAnalysisResult() { return Result; }

  // Keep track of Function frame
  void pushBack(Value *NextVal, Value *CurVal) {
    WorkList.push_back(UseEdge{NextVal, CurVal});
    if (auto *CB = dyn_cast<CallBase>(NextVal)) {
      // todo: do we need current caller CB
      ArgBeforeCB = CallerFrame{.ValEnteringCallBase = CurVal,
                                .CallerCB = CB,
                                .Callee = CB->getParent()->getParent()};
    }
  }

  void offsetValueMapFailure(Value *V) {
    AnalysisFailed = true;
    AnalysisSuccess = false;
    DEBUG(Logger::logs("proteus-pass")
          << "    [PTR use analysis]: Analysis failed due to absence of " << *V
          << " in offset tracking map, this is an internal compiler bug\n");
  }

  void visitStoreInst(StoreInst &SI) {
    // todo: uncomment
    // Value *Stored = SI.getValueOperand();
    // if (!Stored->getType()->isPointerTy())
    //   return;
    Value *StoreBase = SI.getPointerOperand();
    auto StoreSize = getTypeStoreSize(DL, SI.getValueOperand()->getType());
    if (!ValueOffsetMap.contains(StoreBase)) {
      offsetValueMapFailure(StoreBase);
      return;
    }

    if (!StoreSize ||
        !offsetCoveredByRange(ValueOffsetMap[StoreBase], 0, *StoreSize))
      return;
    DEBUG(Logger::logs("proteus-pass")
          << "    Found PTRstore applicable to offset " << ValueOffsetMap[&SI]
          << " Store size = " << *StoreSize << " ; " << SI << "\n");
    AnalysisFailed = false;
    AnalysisSuccess = true;
    Result = {.DominatingWrite = SI.getValueOperand(),
              .Offset = Offset,
              .ChangedRCLayout = std::nullopt};
  }

  void visitLoadInst(LoadInst &LI) {
    if (!LI.getType()->isPointerTy()) {
      AnalysisFailed = true;
      AnalysisSuccess = false;
      return;
    }
    for (User *Usr : LI.users()) {
      pushBack(Usr, &LI);
    }
  }

  void visitCallBase(CallBase &CB) {
    auto OldFrame = ArgBeforeCB;
    DEBUG(Logger::logs("proteus-pass")
          << "    OLD VAL " << *OldFrame.ValEnteringCallBase << "\n");
    Function *F = CB.getCalledFunction();
    Value *ArgToTrack = nullptr;
    for (size_t ArgI = 0; ArgI < F->arg_size(); ++ArgI) {
      DEBUG(Logger::logs("proteus-pass") << "    ARG " << ArgI << " VAL "
                                         << *CB.getArgOperand(ArgI) << "\n");
      if (CB.getArgOperand(ArgI) == ArgBeforeCB.ValEnteringCallBase)
        ArgToTrack = F->getArg(ArgI);
    }
    DEBUG(Logger::logs("proteus-pass") << "    Beginning analysis within "
                                       << *CB.getCalledFunction() << "\n");
    // Propagate the offset across the call boundary.
    auto *DefBeforeCB = getLastDef();
    if (!ValueOffsetMap.contains(DefBeforeCB)) {
      offsetValueMapFailure(DefBeforeCB);
      return;
    }
    ValueOffsetMap[ArgToTrack] = ValueOffsetMap[DefBeforeCB];
    DEBUG(Logger::logs("proteus-pass")
          << "    Looking at uses of " << *ArgToTrack << "\n");
    // Check all uses across the call boundary
    for (User *Usr : ArgToTrack->users()) {
      // todo add check if its a ptr, continue if not.
      pushBack(Usr, ArgToTrack);
    }
  }

  void visitGetElementPtrInst(GetElementPtrInst &GEP) {
    // We don't want to use GetPointerBaseWithConstantOffset here.
    // We actually don't want the true "pointer base" here.  I.E. if we are
    // analyzing the dominating store to %4 = addrspacecast ptr addrspace(5) %3
    // to ptr where %3 = alloca %class.anon.1, align 8, addrspace(5)
    // GetPointerBaseWithConstantOffset gets us 3, which we (a) don't know about
    // and (b) don't care about, we just care about the SSA to %4.
    APInt StepOffset(DL.getIndexTypeSizeInBits(GEP.getType()), 0);
    if (!GEP.accumulateConstantOffset(DL, StepOffset)) {
      AnalysisFailed = true;
      AnalysisSuccess = false;
      return;
    }

    int64_t GEPOffset = StepOffset.getSExtValue();
    DEBUG(Logger::logs("proteus-pass")
          << "    " << "Computed GEP offset " << GEPOffset << "\n");
    Value *GEPBase = GEP.getPointerOperand();
    if (!GEPBase)
      return;

    if (!ValueOffsetMap.contains(GEPBase)) {
      offsetValueMapFailure(GEPBase);
      return;
    }

    auto ResultSize = getTypeStoreSize(DL, GEP.getResultElementType());
    DEBUG(Logger::logs("proteus-pass")
          << "    GEP size = " << *ResultSize << "\n");
    if (ResultSize &&
        !offsetCoveredByRange(ValueOffsetMap[GEPBase], GEPOffset, *ResultSize))
      return;
    DEBUG(Logger::logs("proteus-pass")
          << "    Found GEP applicable to offset=" << ValueOffsetMap[GEPBase]
          << " ; " << GEP << "\n");
    // We found a GEP, now we need to track the GEP itself, so the TargetOffset
    // is now zero again
    ValueOffsetMap[&GEP] = ValueOffsetMap[GEPBase] - GEPOffset;
    Offset += GEPOffset;
    DEBUG(Logger::logs("proteus-pass")
          << "    " << "Setting map K " << GEP << " : " << ValueOffsetMap[&GEP]
          << "\n");
    for (auto *User : GEP.users())
      if (!Seen.contains(User))
        pushBack(User, &GEP);
  }

  // todo: these three methods may need to be changed to find a dominating store
  // particularly for the case of mutable lambdas.
  void visitAllocaInst(AllocaInst &Alloca) {
    if (Def) {
      AnalysisFailed = true;
      AnalysisSuccess = false;
      DEBUG(Logger::logs("proteus-pass")
            << "    Dominating use analysis somehow reached AllocaInst from "
               "non-null def\n");
      return;
    }

    ValueOffsetMap[&Alloca] = Offset;
    for (auto *User : Alloca.users())
      if (!Seen.contains(User))
        pushBack(User, &Alloca);
  }

  // TODO(bowen) come up with a unit test for an analysis starting with
  // a BC
  void visitBitCastInst(BitCastInst &BC) {
    // If the last Def is nullptr, we have just begun the use analysis.
    // In this case, respect the constructor's offset for the pointer operand.
    if (!Def)
      ValueOffsetMap[BC.getOperand(0)] = Offset;
    // AddrSpaceCast does not change the offset we track.
    ValueOffsetMap[&BC] = ValueOffsetMap[BC.getOperand(0)];
    for (auto *User : BC.users())
      if (!Seen.contains(User))
        pushBack(User, &BC);
  }

  void visitAddrSpaceCastInst(AddrSpaceCastInst &ASC) {
    // The constructor automatically populates the map with ASC's offset
    // if its not present we need to rely on the pointer operand's offset
    if (!ValueOffsetMap.contains(&ASC)) {
      if (!ValueOffsetMap.contains(ASC.getPointerOperand())) {
        offsetValueMapFailure(ASC.getPointerOperand());
        return;
      }
      // AddrSpaceCast does not change the offset we track.
      ValueOffsetMap[&ASC] = ValueOffsetMap[ASC.getPointerOperand()];
    }
    DEBUG(Logger::logs("proteus-pass")
          << "    [PTR use analysis]: Setting offset " << ASC << " = "
          << ValueOffsetMap[&ASC]);

    for (auto *User : ASC.users())
      if (!Seen.contains(User)) {
        pushBack(User, &ASC);
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
      DEBUG(Logger::logs("proteus-pass")
            << "  [PTR use analysis]: Failure due to nullptr dst/src " << "\n");
      AnalysisFailed = true;
      return;
    }

    DEBUG(Logger::logs("proteus-pass")
          << "  [PTR use analysis]: Completed instrinsic analysis " << "\n");
    Offset = Offset - DstOff + SrcOff;
    AnalysisSuccess = true;
    AnalysisFailed = false;
    Result = {.DominatingWrite = SrcBase,
              .Offset = Offset,
              .ChangedRCLayout = std::nullopt};
  }
};

inline std::optional<LambdaPtrUseAnalysis>
getDominatingUse(const DataLayout &DL, Value *ValueNeedingAnalysis,
                 Value *SeenUse, int64_t TargetOffset) {
  DEBUG(Logger::logs("proteus-pass")
        << "Beginning PtrUse analysis with offset = " << TargetOffset << "\n");

  LambdaInstUseVisitor Visitor(ValueNeedingAnalysis, SeenUse, DL, TargetOffset);
  // Analysis loop
  while (!Visitor.empty() && !Visitor.success() && !Visitor.failed()) {
    auto *V = Visitor.popBack().CurVal;
    // Prevent loops/infinite recursion
    if (Visitor.seen(V))
      continue;
    Visitor.markAsSeen(V);
    DEBUG(Logger::logs("proteus-pass")
          << "  [PTR use analysis]: Visiting ptr use " << *V << "\n");
    // Analyze the instruction
    if (auto *I = dyn_cast<Instruction>(V))
      Visitor.visit(*I);
    else
      continue;
  }
  if (!Visitor.success() || Visitor.failed()) {
    DEBUG(
        Logger::logs("proteus-pass")
        << "  [PTR use analysis] [WARNING]: Dominating use analysis FAILED for "
        << *ValueNeedingAnalysis << " <-- " << *SeenUse << "\n");
    return std::nullopt;
  }
  LambdaPtrUseAnalysis Info = Visitor.getAnalysisResult();
  if (!Info.DominatingWrite)
    return std::nullopt;
  DEBUG(Logger::logs("proteus-pass")
        << "  [PTR USE ANALYSIS]: Computed offset " << Info.Offset << "\n");
  return Info;

  return std::nullopt;
}
} // namespace proteus

#endif
