#ifndef PROTEUS_KERNELARGVISITOR_H
#define PROTEUS_KERNELARGVISITOR_H

#include <alloca.h>
#include <cstdint>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/Hashing.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/InstVisitor.h>
#include <llvm/IR/IntrinsicInst.h>

#include <memory>
#include <optional>

namespace proteus {
using namespace llvm;

struct FnMemCtx {
  llvm::DominatorTree DT;

  FnMemCtx(llvm::Function &F) : DT(F) {}
};

static void
findAnnotatedFunctions(llvm::Module &M, llvm::StringRef Wanted,
                       llvm::SmallVectorImpl<
                           std::pair<llvm::Function *, std::uint64_t>> &Out) {
  auto *GA = M.getGlobalVariable("llvm.global.annotations");
  if (!GA)
    return;

  auto *CA = llvm::dyn_cast<llvm::ConstantArray>(GA->getOperand(0));
  if (!CA)
    return;

  llvm::SmallDenseMap<llvm::Function *, std::uint64_t, 32> Seen;

  for (llvm::Value *Elt : CA->operands()) {
    auto *CS = llvm::dyn_cast<llvm::ConstantStruct>(Elt);
    if (!CS || CS->getNumOperands() < 5)
      continue;

    auto *F =
        llvm::dyn_cast<llvm::Function>(CS->getOperand(0)->stripPointerCasts());
    if (!F)
      continue;

    llvm::StringRef Ann;
    if (!llvm::getConstantStringInfo(CS->getOperand(1)->stripPointerCasts(),
                                     Ann))
      continue;
    if (Ann != Wanted)
      continue;

    std::uint64_t Id = 0;
    llvm::Value *ArgsPtr = CS->getOperand(4)->stripPointerCasts();
    if (llvm::isa<llvm::ConstantPointerNull>(ArgsPtr))
      continue;

    auto *ArgsGV = llvm::dyn_cast<llvm::GlobalVariable>(ArgsPtr);
    if (!ArgsGV)
      continue;

    auto *ArgsInit = ArgsGV->getInitializer();
    auto *ArgsCS = llvm::dyn_cast<llvm::ConstantStruct>(ArgsInit);
    if (!ArgsCS || ArgsCS->getNumOperands() < 1)
      continue;

    auto *CI = llvm::dyn_cast<llvm::ConstantInt>(ArgsCS->getOperand(0));
    if (!CI)
      continue;

    Id = CI->getZExtValue();

    if (Seen.try_emplace(F, Id).second)
      Out.emplace_back(F, Id);
  }
}

class LambdaArgVisitor : public InstVisitor<LambdaArgVisitor> {
private:
  struct WorkItem {
    Value *V = nullptr;
    Instruction *UseSite = nullptr;
    int64_t Offset = 0;
  };

  struct WorkKey {
    const Value *V = nullptr;
    const Instruction *UseSite = nullptr;
    int64_t Offset = 0;
  };

  struct WorkKeyInfo {
    static inline WorkKey getEmptyKey() {
      return {DenseMapInfo<const Value *>::getEmptyKey(),
              DenseMapInfo<const Instruction *>::getEmptyKey(), 0};
    }

    static inline WorkKey getTombstoneKey() {
      return {DenseMapInfo<const Value *>::getTombstoneKey(),
              DenseMapInfo<const Instruction *>::getTombstoneKey(), 0};
    }

    static unsigned getHashValue(const WorkKey &Key) {
      return static_cast<unsigned>(
          hash_combine(Key.V, Key.UseSite, Key.Offset));
    }

    static bool isEqual(const WorkKey &LHS, const WorkKey &RHS) {
      return LHS.V == RHS.V && LHS.UseSite == RHS.UseSite &&
             LHS.Offset == RHS.Offset;
    }
  };

  enum class WriterKind { Store, MemTransfer, Unsupported };

  struct WriterCandidate {
    WriterKind Kind = WriterKind::Unsupported;
    Instruction *Inst = nullptr;
    Value *NextValue = nullptr;
    int64_t NextOffset = 0;
  };

  CallBase *LambdaCB;
  const DataLayout &DL;
  SmallVector<WorkItem> WorkList;
  DenseSet<WorkKey, WorkKeyInfo> Seen;
  int64_t CurrentOffset = 0;
  int64_t ResultOffset = 0;
  Instruction *CurrentUseSite = nullptr;
  uint32_t KernelArg = 0;
  bool AnalysisSuccess = false;
  bool AnalysisFailed = false;
  DenseMap<Function *, std::unique_ptr<FnMemCtx>> &FunctionAnalysisCache;

  FnMemCtx &getFnMemCtx(Function *F) {
    auto &Entry = FunctionAnalysisCache[F];
    if (!Entry)
      Entry = std::make_unique<FnMemCtx>(*F);
    return *Entry;
  }

  void pushWork(Value *V, Instruction *UseSite, int64_t Offset) {
    WorkList.push_back({V, UseSite, Offset});
  }

  static bool isIgnoredIntrinsic(const IntrinsicInst &II) {
    switch (II.getIntrinsicID()) {
    case Intrinsic::dbg_declare:
    case Intrinsic::dbg_value:
    case Intrinsic::lifetime_start:
    case Intrinsic::lifetime_end:
    case Intrinsic::assume:
      return true;
    default:
      return false;
    }
  }

  Value *getPointerBase(Value *Ptr, int64_t &BaseOffset) const {
    BaseOffset = 0;
    if (Value *Base = GetPointerBaseWithConstantOffset(Ptr, BaseOffset, DL))
      return Base->stripPointerCasts();
    return Ptr ? Ptr->stripPointerCasts() : nullptr;
  }

  bool normalizeTrackedPointer(Value *Ptr, Value *&Root,
                               int64_t &TrackedOffset) const {
    int64_t BaseOffset = 0;
    Root = getPointerBase(Ptr, BaseOffset);
    if (!Root)
      return false;
    TrackedOffset = CurrentOffset + BaseOffset;
    return true;
  }

  std::optional<uint64_t> getFixedStoreSize(Type *Ty) const {
    TypeSize Size = DL.getTypeStoreSize(Ty);
    if (Size.isScalable())
      return std::nullopt;
    return Size.getFixedValue();
  }

  static bool trackedByteIsWithin(int64_t TrackedOffset, int64_t WriteOffset,
                                  uint64_t WriteSize) {
    if (TrackedOffset < WriteOffset)
      return false;
    return static_cast<uint64_t>(TrackedOffset - WriteOffset) < WriteSize;
  }

  bool isBeforeUse(Instruction &I) {
    if (!CurrentUseSite || &I == CurrentUseSite)
      return false;
    if (I.getFunction() != CurrentUseSite->getFunction())
      return false;
    if (I.getParent() == CurrentUseSite->getParent())
      return I.comesBefore(CurrentUseSite);
    return getFnMemCtx(I.getFunction()).DT.dominates(&I, CurrentUseSite);
  }

  bool isLaterCandidate(Instruction &LHS, Instruction &RHS) {
    if (&LHS == &RHS || LHS.getFunction() != RHS.getFunction())
      return false;
    if (LHS.getParent() == RHS.getParent())
      return RHS.comesBefore(&LHS);
    return getFnMemCtx(LHS.getFunction()).DT.dominates(&RHS, &LHS);
  }

  bool pointerUsesTrackedRoot(Value *Ptr, Value *Root,
                              int64_t TrackedOffset) const {
    if (!Ptr || !Ptr->getType()->isPointerTy())
      return false;

    int64_t PtrOffset = 0;
    Value *PtrBase = getPointerBase(Ptr, PtrOffset);
    if (!PtrBase || PtrBase != Root)
      return false;

    return PtrOffset <= TrackedOffset;
  }

  bool classifyWriter(Instruction &I, Value *Root, int64_t TrackedOffset,
                      WriterCandidate &Candidate) {
    if (auto *SI = dyn_cast<StoreInst>(&I)) {
      int64_t StoreOffset = 0;
      Value *StoreBase = getPointerBase(SI->getPointerOperand(), StoreOffset);
      if (!StoreBase || StoreBase != Root)
        return false;

      auto StoreSize = getFixedStoreSize(SI->getValueOperand()->getType());
      if (!StoreSize ||
          !trackedByteIsWithin(TrackedOffset, StoreOffset, *StoreSize))
        return false;

      Candidate.Kind = WriterKind::Store;
      Candidate.Inst = SI;
      Candidate.NextValue = SI->getValueOperand();
      Candidate.NextOffset = TrackedOffset - StoreOffset;
      return true;
    }

    if (auto *MT = dyn_cast<MemTransferInst>(&I)) {
      int64_t DstOffset = 0;
      int64_t SrcOffset = 0;
      Value *DstBase = getPointerBase(MT->getRawDest(), DstOffset);
      Value *SrcBase = getPointerBase(MT->getRawSource(), SrcOffset);
      if (!DstBase || !SrcBase || DstBase != Root)
        return false;

      if (auto *LenC = dyn_cast<ConstantInt>(MT->getLength())) {
        uint64_t Len = LenC->getZExtValue();
        if (!trackedByteIsWithin(TrackedOffset, DstOffset, Len))
          return false;

        Candidate.Kind = WriterKind::MemTransfer;
        Candidate.Inst = MT;
        Candidate.NextValue = SrcBase;
        Candidate.NextOffset = TrackedOffset - DstOffset + SrcOffset;
        return true;
      }

      if (DstOffset <= TrackedOffset) {
        Candidate.Kind = WriterKind::Unsupported;
        Candidate.Inst = MT;
        return true;
      }
      return false;
    }

    if (auto *MS = dyn_cast<MemSetInst>(&I)) {
      int64_t DstOffset = 0;
      Value *DstBase = getPointerBase(MS->getRawDest(), DstOffset);
      if (!DstBase || DstBase != Root)
        return false;

      if (auto *LenC = dyn_cast<ConstantInt>(MS->getLength())) {
        uint64_t Len = LenC->getZExtValue();
        if (!trackedByteIsWithin(TrackedOffset, DstOffset, Len))
          return false;
      } else if (DstOffset > TrackedOffset) {
        return false;
      }

      Candidate.Kind = WriterKind::Unsupported;
      Candidate.Inst = MS;
      return true;
    }

    if (auto *RMW = dyn_cast<AtomicRMWInst>(&I)) {
      int64_t WriteOffset = 0;
      Value *WriteBase = getPointerBase(RMW->getPointerOperand(), WriteOffset);
      auto WriteSize = getFixedStoreSize(RMW->getValOperand()->getType());
      if (!WriteBase || WriteBase != Root || !WriteSize ||
          !trackedByteIsWithin(TrackedOffset, WriteOffset, *WriteSize))
        return false;

      Candidate.Kind = WriterKind::Unsupported;
      Candidate.Inst = RMW;
      return true;
    }

    if (auto *CX = dyn_cast<AtomicCmpXchgInst>(&I)) {
      int64_t WriteOffset = 0;
      Value *WriteBase = getPointerBase(CX->getPointerOperand(), WriteOffset);
      auto WriteSize = getFixedStoreSize(CX->getNewValOperand()->getType());
      if (!WriteBase || WriteBase != Root || !WriteSize ||
          !trackedByteIsWithin(TrackedOffset, WriteOffset, *WriteSize))
        return false;

      Candidate.Kind = WriterKind::Unsupported;
      Candidate.Inst = CX;
      return true;
    }

    if (auto *CB = dyn_cast<CallBase>(&I)) {
      if (&I == CurrentUseSite || !CB->mayWriteToMemory())
        return false;

      if (auto *II = dyn_cast<IntrinsicInst>(CB)) {
        if (isIgnoredIntrinsic(*II))
          return false;
      }

      for (Use &Arg : CB->args()) {
        if (!pointerUsesTrackedRoot(Arg.get(), Root, TrackedOffset))
          continue;

        Candidate.Kind = WriterKind::Unsupported;
        Candidate.Inst = CB;
        return true;
      }
    }

    return false;
  }

  bool followDominatingWriter(Value *Ptr, Value *FallbackPtr = nullptr) {
    Value *Root = nullptr;
    int64_t TrackedOffset = 0;
    if (!normalizeTrackedPointer(Ptr, Root, TrackedOffset)) {
      if (FallbackPtr) {
        pushWork(FallbackPtr, CurrentUseSite, CurrentOffset);
        return true;
      }
      AnalysisFailed = true;
      return false;
    }

    Function *F = CurrentUseSite ? CurrentUseSite->getFunction() : nullptr;
    if (!F) {
      AnalysisFailed = true;
      return false;
    }

    bool FoundWriter = false;
    WriterCandidate Best;

    for (BasicBlock &BB : *F) {
      for (Instruction &I : BB) {
        if (!isBeforeUse(I))
          continue;

        WriterCandidate Candidate;
        if (!classifyWriter(I, Root, TrackedOffset, Candidate))
          continue;

        if (!FoundWriter) {
          Best = Candidate;
          FoundWriter = true;
          continue;
        }

        if (isLaterCandidate(I, *Best.Inst)) {
          Best = Candidate;
          continue;
        }

        if (isLaterCandidate(*Best.Inst, I))
          continue;

        AnalysisFailed = true;
        return false;
      }
    }

    if (!FoundWriter) {
      if (FallbackPtr) {
        pushWork(FallbackPtr, CurrentUseSite, CurrentOffset);
        return true;
      }
      AnalysisFailed = true;
      return false;
    }

    if (Best.Kind == WriterKind::Unsupported) {
      AnalysisFailed = true;
      return false;
    }

    pushWork(Best.NextValue, Best.Inst, Best.NextOffset);
    return true;
  }

public:
  LambdaArgVisitor(
      CallBase *LambdaCB_, Module &M,
      DenseMap<Function *, std::unique_ptr<FnMemCtx>> &Cache_)
      : LambdaCB(LambdaCB_), DL(M.getDataLayout()),
        FunctionAnalysisCache(Cache_) {
    auto *ClosurePtr = LambdaCB->getArgOperand(0);
    pushWork(ClosurePtr, LambdaCB, 0);
  }

  void visitStoreInst(StoreInst &SI) {
    pushWork(SI.getValueOperand(), &SI, CurrentOffset);
  }

  void visitCallBase(CallBase &) {}

  void visitLoadInst(LoadInst &LI) {
    if (!LI.getType()->isPointerTy()) {
      AnalysisFailed = true;
      return;
    }

    int64_t LoadOff = 0;
    Value *LoadBase = getPointerBase(LI.getPointerOperand(), LoadOff);
    if (!LoadBase) {
      AnalysisFailed = true;
      return;
    }

    SmallVector<Value *, 8> PtrWorkList;
    SmallPtrSet<Value *, 16> LocalSeen;
    PtrWorkList.push_back(LoadBase);

    Value *CommonStoredVal = nullptr;
    bool FoundStore = false;

    while (!PtrWorkList.empty()) {
      Value *Cur = PtrWorkList.pop_back_val();
      if (!LocalSeen.insert(Cur).second)
        continue;

      for (User *U : Cur->users()) {
        if (auto *SI = dyn_cast<StoreInst>(U)) {
          int64_t StoreOff = 0;
          Value *StoreBase = getPointerBase(SI->getPointerOperand(), StoreOff);
          if (!StoreBase || StoreBase != LoadBase || StoreOff != LoadOff)
            continue;

          Value *V = SI->getValueOperand();
          if (!V->getType()->isPointerTy())
            continue;

          V = V->stripPointerCasts();
          if (!FoundStore) {
            CommonStoredVal = V;
            FoundStore = true;
          } else if (CommonStoredVal != V) {
            AnalysisFailed = true;
            return;
          }
          continue;
        }

        if (isa<GetElementPtrInst>(U) || isa<BitCastInst>(U) ||
            isa<AddrSpaceCastInst>(U) || isa<PHINode>(U) ||
            isa<SelectInst>(U)) {
          PtrWorkList.push_back(cast<Value>(U));
          continue;
        }

        if (auto *II = dyn_cast<IntrinsicInst>(U)) {
          if (isIgnoredIntrinsic(*II))
            continue;
        }
      }
    }

    if (!FoundStore) {
      AnalysisFailed = true;
      return;
    }

    pushWork(CommonStoredVal, &LI, CurrentOffset);
  }

  auto back() { return WorkList.back(); }
  void popBack() { WorkList.pop_back(); }

  bool seen(const WorkItem &Item) {
    return Seen.contains({Item.V, Item.UseSite, Item.Offset});
  }

  void markAsSeen(const WorkItem &Item) {
    Seen.insert({Item.V, Item.UseSite, Item.Offset});
  }

  bool empty() { return WorkList.empty(); }
  bool success() { return AnalysisSuccess; }
  bool failed() { return AnalysisFailed; }

  void setCurrentItem(const WorkItem &Item) {
    CurrentUseSite = Item.UseSite;
    CurrentOffset = Item.Offset;
  }

  auto getKernelArgAndOffset() {
    return std::make_pair(KernelArg, ResultOffset);
  }

  void visitGetElementPtrInst(GetElementPtrInst &GEP) {
    int64_t GEPOffset = 0;
    if (!GetPointerBaseWithConstantOffset(&GEP, GEPOffset, DL)) {
      AnalysisFailed = true;
      return;
    }
    pushWork(GEP.getPointerOperand(), CurrentUseSite,
             CurrentOffset + GEPOffset);
  }

  void visitAllocaInst(AllocaInst &Alloca) { followDominatingWriter(&Alloca); }

  void visitBitCastInst(BitCastInst &BC) {
    followDominatingWriter(&BC, BC.getOperand(0));
  }

  void visitAddrSpaceCastInst(AddrSpaceCastInst &ASC) {
    followDominatingWriter(&ASC, ASC.getOperand(0));
  }

  void visitMemIntrinsic(MemIntrinsic &I) {
    auto *MT = dyn_cast<MemTransferInst>(&I);
    if (!MT) {
      AnalysisFailed = true;
      return;
    }

    int64_t DstOff = 0;
    int64_t SrcOff = 0;
    Value *DstBase = getPointerBase(MT->getRawDest(), DstOff);
    Value *SrcBase = getPointerBase(MT->getRawSource(), SrcOff);
    if (!DstBase || !SrcBase) {
      AnalysisFailed = true;
      return;
    }

    if (auto *LenC = dyn_cast<ConstantInt>(MT->getLength())) {
      uint64_t Len = LenC->getZExtValue();
      if (!trackedByteIsWithin(CurrentOffset, DstOff, Len)) {
        AnalysisFailed = true;
        return;
      }
    } else {
      AnalysisFailed = true;
      return;
    }

    pushWork(SrcBase, &I, CurrentOffset - DstOff + SrcOff);
  }

  void visitIntrinsicInst(IntrinsicInst &II) {
    if (isIgnoredIntrinsic(II))
      return;
    AnalysisFailed = true;
  }

  void visitArgument(Argument &A) {
    Function *F = A.getParent();
    auto ArgNum = A.getArgNo();

    if (F->getCallingConv() == CallingConv::AMDGPU_KERNEL ||
        F->getCallingConv() == CallingConv::PTX_Kernel) {
      AnalysisSuccess = true;
      KernelArg = ArgNum;
      ResultOffset = CurrentOffset;
      return;
    }

    for (User *U : F->users()) {
      auto *CB = dyn_cast<CallBase>(U);
      if (!CB)
        continue;
      pushWork(CB->getArgOperand(ArgNum), CB, CurrentOffset);
    }
  }

  void visitInstruction(Instruction &) {
    AnalysisFailed = true;
    return;
  }

  void visitPHINode(PHINode &P) {
    if (P.getNumIncomingValues() == 0) {
      AnalysisFailed = true;
      return;
    }

    Value *CommonBase = nullptr;
    int64_t CommonIncOff = 0;
    bool First = true;

    for (Value *Inc : P.incoming_values()) {
      int64_t IncOff = 0;
      Value *Base = getPointerBase(Inc, IncOff);
      if (!Base) {
        AnalysisFailed = true;
        return;
      }

      if (First) {
        CommonBase = Base;
        CommonIncOff = IncOff;
        First = false;
        continue;
      }

      if (Base != CommonBase || IncOff != CommonIncOff) {
        AnalysisFailed = true;
        return;
      }
    }

    pushWork(CommonBase, CurrentUseSite, CurrentOffset + CommonIncOff);
  }

  void visitSelectInst(SelectInst &S) {
    int64_t TOff = 0;
    int64_t FOff = 0;

    Value *TBase = getPointerBase(S.getTrueValue(), TOff);
    Value *FBase = getPointerBase(S.getFalseValue(), FOff);
    if (!TBase || !FBase) {
      AnalysisFailed = true;
      return;
    }

    if (TBase != FBase || TOff != FOff) {
      AnalysisFailed = true;
      return;
    }

    pushWork(TBase, CurrentUseSite, CurrentOffset + TOff);
  }
};

inline bool analyzeLambdaUses(
    llvm::Module &M,
    DenseMap<CallBase *, std::pair<uint32_t, int64_t>> &CallBaseToArgOffset,
    const SmallVector<CallBase *> &CBToAnalyze,
    DenseMap<Function *, std::unique_ptr<FnMemCtx>> &FunctionAnalysisCache) {
  for (auto *FunctorCB : CBToAnalyze) {
    LambdaArgVisitor Visitor(FunctorCB, M, FunctionAnalysisCache);
    Value *LastValue = nullptr;
    Instruction *LastUseSite = nullptr;
    int64_t LastOffset = 0;
    while (!Visitor.empty() && !Visitor.success() && !Visitor.failed()) {
      auto Item = Visitor.back();
      Visitor.popBack();

      if (Visitor.seen(Item))
        continue;
      Visitor.markAsSeen(Item);
      Visitor.setCurrentItem(Item);
      LastValue = Item.V;
      LastUseSite = Item.UseSite;
      LastOffset = Item.Offset;

      auto *V = Item.V;
      if (auto *I = dyn_cast<Instruction>(V))
        Visitor.visit(*I);
      else if (auto *A = dyn_cast<Argument>(V))
        Visitor.visitArgument(*A);
      else
        continue;
    }

    if (!Visitor.success() || Visitor.failed()) {
      errs() << "[proteus][debug] analyzeLambdaUses failed for callsite "
             << *FunctorCB << "\n";
      if (LastValue)
        errs() << "  last value: " << *LastValue << "\n";
      if (LastUseSite)
        errs() << "  last use-site: " << *LastUseSite << "\n";
      errs() << "  last offset: " << LastOffset << "\n";
      return false;
    }
    CallBaseToArgOffset[FunctorCB] = Visitor.getKernelArgAndOffset();
  }
  return true;
}
} // namespace proteus

#endif
