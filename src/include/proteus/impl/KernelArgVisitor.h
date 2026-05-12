#ifndef PROTEUS_KERNELARGVISITOR_H
#define PROTEUS_KERNELARGVISITOR_H

#include "proteus/CompilerInterfaceTypes.h"
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
#include <llvm/IR/Function.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Type.h>
#include <memory>
#include <optional>

namespace proteus {
using namespace llvm;

struct LambdaKernelArgAnalysis {
  Function *KernelFunction = nullptr;
  uint32_t KernelArgIndex = 0;
  int64_t Offset = 0;
  // Sometimes instructions like ptrtoint --> inttoptr change the layout of
  // the kernel args.
  std::optional<RuntimeConstantType> ChangedRCLayout = std::nullopt;
};

struct FnMemCtx {
  // llvm::DominatorTree DT;

  FnMemCtx(llvm::Function &) {}
};

class LambdaArgVisitor : public InstVisitor<LambdaArgVisitor> {
private:
  CallBase *LambdaCB;
  const DataLayout &DL;
  SmallVector<Value *> WorkList;
  SmallDenseSet<Value *> Seen;
  int64_t Offset;
  uint32_t KernelArg = 0;
  Function *KernelFunction = nullptr;
  // instructions like inttoptr can change the runtimeconstant type
  // we to read from the Blob
  std::optional<RuntimeConstantType> ChangedRC = std::nullopt;
  bool AnalysisSuccess = false;
  bool AnalysisFailed = false;

  // Constructor used for cloning and merging branches of phi node analysis
  LambdaArgVisitor(Value *Start, int64_t Off, const DataLayout &_DL)
      : DL(_DL), Offset(Off) {
    WorkList.push_back(Start);
  }

public:
  LambdaKernelArgAnalysis getKernelArgInfo() {
    return LambdaKernelArgAnalysis{KernelFunction, KernelArg, Offset,
                                   ChangedRC};
  }
  auto back() { return WorkList.back(); }
  void popBack() { WorkList.pop_back(); }
  bool seen(Value *Val) { return Seen.contains(Val); }
  void markAsSeen(Value *Val) { Seen.insert(Val); }
  bool empty() { return WorkList.empty(); }
  bool success() { return AnalysisSuccess; }
  bool failed() { return AnalysisFailed; }

private:
  inline std::optional<LambdaKernelArgAnalysis>
  cloneAndAnalyze(Value *Start, int64_t StartOffset) {
    LambdaArgVisitor Visitor(Start, StartOffset, DL);
    while (!Visitor.empty() && !Visitor.success() && !Visitor.failed()) {
      auto *V = (Visitor.back());
      Visitor.popBack();
      // Prevent loops/infinite recursion
      if (Visitor.seen(V))
        continue;
      Visitor.markAsSeen(V);
      // Analyze the instruction
      if (auto *I = dyn_cast<Instruction>(V))
        Visitor.visit(*I);
      else if (auto *A = dyn_cast<Argument>(V))
        Visitor.visitArgument(*A);
      else
        continue;
    }
    if (!Visitor.success() || Visitor.failed())
      return std::nullopt;
    return Visitor.getKernelArgInfo();
  }

public:
  LambdaArgVisitor(CallBase *LambdaCB_, Module &M)
      : LambdaCB(LambdaCB_), DL(M.getDataLayout()), Offset(0) {
    auto *ClosurePtr = LambdaCB->getArgOperand(0);
    WorkList.push_back(ClosurePtr);
  }
  void visitStoreInst(StoreInst &SI) {
    WorkList.push_back(SI.getValueOperand());
  }
  void visitCallBase(CallBase &) {}
  void visitIntToPtr(IntToPtrInst &ITP) {
    auto *IntegerVal = ITP.getOperand(0);
    auto *Ptr = dyn_cast<PtrToIntInst>(IntegerVal);
    if (!Ptr) {
      AnalysisSuccess = false;
      AnalysisFailed = true;
      return;
    }
    ChangedRC = convertTypeToRuntimeConstantType(Ptr->getType());

    WorkList.push_back(Ptr->getPointerOperand());
  }

  void visitLoadInst(LoadInst &LI) {
    if (!LI.getType()->isPointerTy()) {
      AnalysisFailed = true;
      return;
    }

    int64_t LoadOff = 0;
    Value *LoadBase =
        GetPointerBaseWithConstantOffset(LI.getPointerOperand(), LoadOff, DL);
    if (!LoadBase) {
      AnalysisFailed = true;
      return;
    }
    LoadBase = LoadBase->stripPointerCasts();

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
          Value *StoreBase = GetPointerBaseWithConstantOffset(
              SI->getPointerOperand(), StoreOff, DL);
          if (!StoreBase)
            continue;
          StoreBase = StoreBase->stripPointerCasts();

          if (StoreBase != LoadBase || StoreOff != LoadOff)
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
          switch (II->getIntrinsicID()) {
          case Intrinsic::dbg_declare:
          case Intrinsic::dbg_value:
          case Intrinsic::lifetime_start:
          case Intrinsic::lifetime_end:
            continue;
          default:
            break;
          }
        }
      }
    }

    if (!FoundStore) {
      AnalysisFailed = true;
      return;
    }

    WorkList.push_back(CommonStoredVal);
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

  void visitArgument(Argument &A) {
    Function *F = A.getParent();
    auto ArgNum = A.getArgNo();
    // termination case:  we have reached the parent calling kernel
    // todo: we could just pass in the kernel pointer here and check equality
    if (F->getCallingConv() == CallingConv::AMDGPU_KERNEL ||
        F->getCallingConv() == CallingConv::PTX_Kernel) {
      AnalysisSuccess = true;
      KernelArg = ArgNum;
      KernelFunction = F;
      return;
    }

    for (User *U : F->users()) {
      auto *CB = dyn_cast<CallBase>(U);
      if (!CB)
        continue;
      WorkList.push_back(CB->getArgOperand(ArgNum));
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

    auto FirstAnalysis = cloneAndAnalyze(P.getIncomingValue(0), Offset);
    if (!FirstAnalysis) {
      AnalysisFailed = true;
      AnalysisSuccess = false;
      return;
    }
    auto BaseSlot = FirstAnalysis->KernelArgIndex;
    auto BaseOffset = FirstAnalysis->Offset;

    for (size_t Idx = 1; Idx < P.getNumIncomingValues(); ++Idx) {
      auto Analysis = cloneAndAnalyze(P.getIncomingValue(Idx), Offset);
      if (!Analysis || Analysis->KernelArgIndex != BaseSlot ||
          Analysis->Offset != BaseOffset) {
        AnalysisFailed = true;
        AnalysisSuccess = false;
        return;
      }
    }
    KernelFunction = FirstAnalysis->KernelFunction;
    AnalysisSuccess = true;
    AnalysisFailed = false;
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

inline bool analyzeLambdaUses(
    llvm::Module &M,
    DenseMap<CallBase *, LambdaKernelArgAnalysis> &CallBaseToArgOffset,
    const SmallVector<CallBase *> &CBToAnalyze) {
  for (auto *FunctorCB : CBToAnalyze) {
    LambdaArgVisitor Visitor(FunctorCB, M);
    while (!Visitor.empty() && !Visitor.success() && !Visitor.failed()) {
      auto *V = Visitor.back();
      Visitor.popBack();
      // Prevent loops/infinite recursion
      if (Visitor.seen(V))
        continue;
      Visitor.markAsSeen(V);
      // Analyze the instruction
      if (auto *I = dyn_cast<Instruction>(V))
        Visitor.visit(*I);
      else if (auto *A = dyn_cast<Argument>(V))
        Visitor.visitArgument(*A);
      else
        continue;
    }
    if (!Visitor.success() || Visitor.failed())
      return false;
    LambdaKernelArgAnalysis Info = Visitor.getKernelArgInfo();
    if (!Info.KernelFunction)
      return false;
    CallBaseToArgOffset[FunctorCB] = Info;
  }
  return true;
}
} // namespace proteus

#endif
