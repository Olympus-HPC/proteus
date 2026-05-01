#include <alloca.h>
#include <cstdint>
#include <llvm/Analysis/PtrUseVisitor.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instruction.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>

using namespace llvm;

class LambdaArgVisitor : InstVisitor<LambdaArgVisitor> {
private:
  CallBase* LambdaCB;
  const DataLayout &DL;
  SmallVector<Value*> WorkList;
  SmallDenseSet<Value*> Seen;
  int64_t Offset;
  uint32_t KernelArg = 0;
  bool AnalysisSuccess = true;
public:
LambdaArgVisitor(CallBase* LambdaCB_, Module& M, SmallVector<Value*> WorkList_, SmallDenseSet<Value*> Seen_) : WorkList(WorkList_), Seen(Seen_), DL(M.getDataLayout()){
  WorkList.push_back(LambdaCB_->getArgOperand(0));
}
LambdaArgVisitor(CallBase* LambdaCB_, Module& M) : DL(M.getDataLayout()){
  WorkList.push_back(LambdaCB_->getArgOperand(0));
}
void visitStoreInst (StoreInst &SI) { WorkList.push_back(SI.getPointerOperand()); }
void visitBitCastInst (BitCastInst &BC) { WorkList.push_back(BC.getOperand(0)); }
void visitAddrSpaceCastInst (AddrSpaceCastInst &ASC) { WorkList.push_back(ASC.getPointerOperand() ); }

auto back() { return WorkList.back(); }
void popBack() { WorkList.pop_back(); }
bool seen(Value* Val) { return Seen.contains(Val); }
void markAsSeen(Value* Val) { Seen.insert(Val); }
bool empty() { return WorkList.empty(); }
bool success() { return AnalysisSuccess; }
auto getKernelArgAndOffset() { return std::make_pair(KernelArg, Offset); }

void visitGetElementPtrInst (GetElementPtrInst &GEP) {
  APInt GEPOffset;
  if (!GEP.accumulateConstantOffset(DL, GEPOffset))
    return;
  int64_t GepOff = GEPOffset.getSExtValue(); // bytes
  if (__builtin_add_overflow(Offset, GepOff, &Offset)) {
    AnalysisSuccess = false;
    return;
  }
  WorkList.push_back(GEP.getPointerOperand());
  // Increment the offset computed by this analysis
  Offset += GepOff;
}

void visitAllocaInst (AllocaInst &Alloca) {
  for (User* U : Alloca.users()) {
    auto* MemIntrins = dyn_cast<MemIntrinsic>(U);
    auto* Store = dyn_cast<StoreInst>(U);
    if (!MemIntrins && !Store)
      continue;
    WorkList.push_back(U);
  }
}

void visitMemIntrinsic(MemIntrinsic &I) {
  auto *MT = dyn_cast<MemTransferInst>(&I); // memcpy/memmove
  if (!MT) {
    // memset doesn't preserve any src->dst relationship we can use
    AnalysisSuccess = false;
    return;
  }

  int64_t DstOff = 0, SrcOff = 0;
  Value *DstBase = GetPointerBaseWithConstantOffset(MT->getRawDest(),   DstOff, DL);
  Value *SrcBase = GetPointerBaseWithConstantOffset(MT->getRawSource(), SrcOff, DL);
  if (!DstBase || !SrcBase) {
    AnalysisSuccess = false;
    return;
  }

  // Optional safety: only valid if the tracked byte lies within the copied region.
  if (auto *LenC = dyn_cast<ConstantInt>(MT->getLength())) {
    uint64_t Len = LenC->getZExtValue();
    if (Offset < DstOff || uint64_t(Offset - DstOff) >= Len) {
      AnalysisSuccess = false;
      return;
    }
  }

  WorkList.push_back(SrcBase->stripPointerCasts());
  Offset = Offset - DstOff + SrcOff;
}

void visitIntrinsicInst (IntrinsicInst &II) {
  AnalysisSuccess = false;
  return;
}

void visitArgument(Argument &A) {
  Function* F = A.getParent();
  auto ArgNum = A.getArgNo();
  if (F->hasFnAttribute("amdgpu_kernel")) {
    AnalysisSuccess = true;
    KernelArg = ArgNum;
    return;
  }
  for (User* U : F->users()) {
    auto* CB = dyn_cast<CallBase>(U);
    if (!CB)
      continue;
    WorkList.push_back(CB->getArgOperand(ArgNum));
  }
}

void visitInstruction (Instruction&) {
  AnalysisSuccess = false;
  return;
}

// todo add tests for the below two methods
void visitPHINode(PHINode &P) {
  if (P.getNumIncomingValues() == 0) {
    AnalysisSuccess = false;
    return;
  }

  Value *CommonBase = nullptr;
  int64_t CommonIncOff = 0;
  bool First = true;

  for (Value *Inc : P.incoming_values()) {
    int64_t IncOff = 0;
    Value *Base = GetPointerBaseWithConstantOffset(Inc, IncOff, DL);
    Base = Base->stripPointerCasts();

    if (First) {
      CommonBase = Base;
      CommonIncOff = IncOff;
      First = false;
      continue;
    }

    if (Base != CommonBase || IncOff != CommonIncOff) {
      AnalysisSuccess = false; // would need path-sensitive offsets to proceed
      return;
    }
  }

  WorkList.push_back(CommonBase);
  Offset += CommonIncOff; // phi == CommonBase + CommonIncOff
}

void visitSelectInst(SelectInst &S) {
  int64_t TOff = 0;
  int64_t FOff = 0;

  Value *TBase = GetPointerBaseWithConstantOffset(S.getTrueValue(), TOff, DL);
  Value *FBase = GetPointerBaseWithConstantOffset(S.getFalseValue(), FOff, DL);
  TBase = TBase->stripPointerCasts();
  FBase = FBase->stripPointerCasts();

  if (TBase != FBase || TOff != FOff) {
    AnalysisSuccess = false; // would need path-sensitive offsets to proceed
    return;
  }

  WorkList.push_back(TBase);
  Offset += TOff; // select == TBase + TOff
}

};
