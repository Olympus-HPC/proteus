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

class FunctionMemorySSAResolver {
private:
  struct FunctionAnalyses {
    std::unique_ptr<TargetLibraryInfoImpl> TLII;
    std::unique_ptr<TargetLibraryInfo> TLI;
    std::unique_ptr<AssumptionCache> AC;
    std::unique_ptr<DominatorTree> DT;
    std::unique_ptr<AAResults> AA;
    std::unique_ptr<BasicAAResult> BAA;
    std::unique_ptr<MemorySSA> MSSA;
  };

  DenseMap<Function *, FunctionAnalyses> Cache;
  const DataLayout &DL;

  FunctionAnalyses &getAnalyses(Function &F) {
    auto [It, Inserted] = Cache.try_emplace(&F);
    if (Inserted) {
      It->second.TLII = std::make_unique<TargetLibraryInfoImpl>(
          Triple(F.getParent()->getTargetTriple()));
      It->second.TLI = std::make_unique<TargetLibraryInfo>(*It->second.TLII);
      It->second.AC = std::make_unique<AssumptionCache>(F);
      It->second.DT = std::make_unique<DominatorTree>(F);
      It->second.AA = std::make_unique<AAResults>(*It->second.TLI);
      It->second.BAA = std::make_unique<BasicAAResult>(
          DL, F, *It->second.TLI, *It->second.AC, It->second.DT.get());
      It->second.AA->addAAResult(*It->second.BAA);
      It->second.MSSA = std::make_unique<MemorySSA>(F, It->second.AA.get(),
                                                    It->second.DT.get());
    }
    return It->second;
  }

  std::optional<LambdaPtrUseAnalysis> resolveMemoryAccess(MemoryAccess *MA,
                                                          int64_t Offset) {
    if (!MA)
      return std::nullopt;

    if (auto *MP = dyn_cast<MemoryPhi>(MA)) {
      std::optional<LambdaPtrUseAnalysis> Common;
      for (Use &Incoming : MP->incoming_values()) {
        auto *IncomingMA = dyn_cast<MemoryAccess>(Incoming.get());
        if (!IncomingMA)
          return std::nullopt;
        auto Res = resolveMemoryAccess(IncomingMA, Offset);
        if (!Res)
          return std::nullopt;
        if (!Common) {
          Common = Res;
          continue;
        }
        if (Common->DominatingWrite != Res->DominatingWrite ||
            Common->Offset != Res->Offset ||
            Common->ChangedRCLayout != Res->ChangedRCLayout)
          return std::nullopt;
      }
      return Common;
    }

    auto *MD = dyn_cast<MemoryDef>(MA);
    if (!MD)
      return std::nullopt;

    auto *MemI = dyn_cast_or_null<Instruction>(MD->getMemoryInst());
    if (!MemI)
      return std::nullopt;

    if (auto *SI = dyn_cast<StoreInst>(MemI)) {
      Value *Stored = SI->getValueOperand();
      if (!Stored->getType()->isPointerTy())
        return std::nullopt;
      return LambdaPtrUseAnalysis{Stored, Offset, std::nullopt};
    }

    auto *MT = dyn_cast<MemTransferInst>(MemI);
    if (!MT)
      return std::nullopt;

    int64_t DstOff = 0, SrcOff = 0;
    Value *DstBase =
        GetPointerBaseWithConstantOffset(MT->getRawDest(), DstOff, DL);
    Value *SrcBase =
        GetPointerBaseWithConstantOffset(MT->getRawSource(), SrcOff, DL);
    if (!DstBase || !SrcBase)
      return std::nullopt;

    return LambdaPtrUseAnalysis{SrcBase, Offset - DstOff + SrcOff,
                                std::nullopt};
  }

public:
  explicit FunctionMemorySSAResolver(const DataLayout &Dl) : DL(Dl) {}

  std::optional<LambdaPtrUseAnalysis> resolve(Value *TrackedPtr,
                                              Instruction *UseI) {
    auto Loc = getTrackedPointerLocation(DL, TrackedPtr);
    if (!Loc)
      return std::nullopt;

    Function &F = *UseI->getFunction();
    auto &Analyses = getAnalyses(F);
    auto *MA = Analyses.MSSA->getMemoryAccess(UseI);
    if (!MA)
      return std::nullopt;

    auto *Walker = Analyses.MSSA->getWalker();
    auto *Clobber = Walker->getClobberingMemoryAccess(MA, *Loc);
    return resolveMemoryAccess(Clobber, /*Offset=*/0);
  }
};

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
  SmallVector<Value *> WorkList;
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
    WorkList.push_back(PtrBegin);
    Seen.insert(SeenUse);
    ValueOffsetMap[PtrBegin] = TargetOff;
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
  void pushBack(Value *NextVal, Value *CurVal) {
    WorkList.push_back(NextVal);
    if (auto *CB = dyn_cast<CallBase>(NextVal)) {
      // todo: do we need current caller CB
      ArgBeforeCB = CallerFrame{.ValEnteringCallBase = CurVal,
                                .CallerCB = CB,
                                .Callee = CB->getParent()->getParent()};
    }
  }

  void visitStoreInst(StoreInst &SI) {
    // todo: uncomment
    // Value *Stored = SI.getValueOperand();
    // if (!Stored->getType()->isPointerTy())
    //   return;
    Value *StoreBase = SI.getPointerOperand();
    auto StoreSize = getTypeStoreSize(DL, SI.getValueOperand()->getType());
    if (!ValueOffsetMap.contains(StoreBase)) {
      DEBUG(Logger::logs("proteus-pass")
            << "    Analysis failed due to absence of " << *StoreBase
            << " in offset tracking map, this is an internal compiler bug\n");
      AnalysisFailed = true;
      AnalysisSuccess = false;
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

    for (User *Usr : ArgToTrack->users())
      pushBack(Usr, ArgToTrack);
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
    if (!GEPBase ||
        GEPBase->stripPointerCasts() != TrackedBase->stripPointerCasts())
      return;

    if (!ValueOffsetMap.contains(GEPBase)) {
      DEBUG(Logger::logs("proteus-pass")
            << "  [PTR use analysis]: " << "GEPBase not found in map "
            << *GEPBase << "\n");
      AnalysisFailed = true;
      AnalysisSuccess = false;
      return;
    }

    auto ResultSize = getTypeStoreSize(DL, GEP.getResultElementType());
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
    DEBUG(Logger::logs("proteus-pass") << ASC << "\n");
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

    // Optional safety: only valid if the tracked byte lies within the copied
    // region.
    // if (auto *LenC = dyn_cast<ConstantInt>(MT->getLength())) {
    //   uint64_t Len = LenC->getZExtValue();
    //   if (Offset < DstOff || uint64_t(Offset - DstOff) >= Len) {
    //     DEBUG(Logger::logs("proteus-pass") << "  [PTR use analysis]: Failure
    //     due to bytesize" << "\n");
    //     // AnalysisFailed = true;
    //     // return;
    //   }
    // }
    DEBUG(Logger::logs("proteus-pass")
          << "  [PTR use analysis]: Completed instrinsic analysis " << "\n");
    Offset = Offset - DstOff + SrcOff;
    AnalysisSuccess = true;
    AnalysisFailed = false;
    Result = {.DominatingWrite = SrcBase,
              .Offset = Offset,
              .ChangedRCLayout = std::nullopt};
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

inline std::optional<LambdaPtrUseAnalysis>
getDominatingUse(const DataLayout &DL, Value *ValueNeedingAnalysis,
                 Value *SeenUse, int64_t TargetOffset) {
  DEBUG(Logger::logs("proteus-pass") << "Beginning PtrUse analysis " << "\n");

  // if (auto *I = dyn_cast<Instruction>(SeenUse)) {
  //   FunctionMemorySSAResolver Resolver(DL);
  //   if (auto Res = Resolver.resolve(ValueNeedingAnalysis, I)) {
  //     DEBUG(Logger::logs("proteus-pass")
  //           << "  [PTR use analysis]: MemorySSA resolved "
  //           << *ValueNeedingAnalysis << " at use " << *SeenUse << "\n");
  //     return Res;
  //   }
  // }

  LambdaInstUseVisitor Visitor(ValueNeedingAnalysis, SeenUse, DL, TargetOffset);
  // Analysis loop
  while (!Visitor.empty() && !Visitor.success() && !Visitor.failed()) {
    auto *V = Visitor.back();
    Visitor.popBack();
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
