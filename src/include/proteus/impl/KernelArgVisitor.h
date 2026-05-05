#ifndef PROTEUS_KERNELARGVISITOR_H
#define PROTEUS_KERNELARGVISITOR_H

#include <alloca.h>
#include <cstdint>
#include <llvm/Analysis/PtrUseVisitor.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instruction.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
// #include <llvm/Analysis/MemorySSA.h>
// #include <llvm/Analysis/MemoryLocation.h>
// #include <llvm/Passes/PassBuilder.h>
// #include <llvm/Analysis/MemorySSA.h>
// #include <llvm/Analysis/TargetLibraryInfo.h>
// #include <llvm/Analysis/MemoryLocation.h>
// #include <llvm/Analysis/AliasAnalysis.h>
// #include <llvm/Analysis/AssumptionCache.h>
// #include <llvm/Analysis/BasicAliasAnalysis.h>
// #include <llvm/Analysis/CaptureTracking.h>
#include <memory>


namespace proteus {
using namespace llvm;
struct FnMemCtx {
//   llvm::DominatorTree DT;
//   llvm::AssumptionCache AC;
//   llvm::TargetLibraryInfoImpl TLII;
//   llvm::TargetLibraryInfo TLI;
//   llvm::BasicAAResult BAA;
//   llvm::AAResults AA;
//   llvm::MemorySSA MSSA;
//   llvm::MemorySSAWalker *Walker = nullptr;

  FnMemCtx(llvm::Function &) {}
//       : DT(F),                         // computes dominators
//         AC(F),                         // scans assumptions in F
//         TLII(llvm::Triple(F.getParent()->getTargetTriple())),
//         TLI(TLII, &F),                 // applies nobuiltin attrs, etc
//         BAA(F.getParent()->getDataLayout(), F, TLI, AC, &DT),
//         AA(TLI),
//         MSSA(F, &AA, &DT) {            // builds MemorySSA immediately
//     AA.addAAResult(BAA);
//     Walker = MSSA.getWalker();
//   }
};

static void findAnnotatedFunctions(
      llvm::Module &M, llvm::StringRef Wanted,
      llvm::SmallVectorImpl<std::pair<llvm::Function *, std::uint64_t>> &Out) {
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

      auto *F = llvm::dyn_cast<llvm::Function>(
          CS->getOperand(0)->stripPointerCasts());
      if (!F)
        continue;

      llvm::StringRef Ann;
      if (!llvm::getConstantStringInfo(CS->getOperand(1)->stripPointerCasts(),
                                       Ann))
        continue;
      if (Ann != Wanted)
        continue;

      // Parse the u64 payload (operand 4)
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
	  CallBase* LambdaCB;
	  const DataLayout &DL;
    SmallVector<Value*> WorkList;
    SmallDenseSet<Value*> Seen;
    int64_t Offset;
    uint32_t KernelArg = 0;
	  bool AnalysisSuccess = false;
	  bool AnalysisFailed = false;
	  DenseMap<Function*, std::unique_ptr<FnMemCtx>>& FunctionAnalysisCache;

    // MemoryAccess* findClobberingWriteBeforeCB(CallBase *CB, int ArgNum) {
    //   Function* CallerFunction = CB->getCaller();
    //   if (!FunctionAnalysisCache.contains(CallerFunction))
    //     FunctionAnalysisCache[CallerFunction] = std::make_unique<FnMemCtx>(*CallerFunction);
    //   auto& CachedAnalysis = *FunctionAnalysisCache[CallerFunction];
    //   MemoryAccess *MA = CachedAnalysis.MSSA.getMemoryAccess(CB);
    //   auto *UD = cast<MemoryUseOrDef>(MA);
    //   MemoryAccess *BeforeCB = UD->getDefiningAccess();
    //   MemoryLocation Loc = MemoryLocation::getForArgument(CB, /*ArgIdx=*/ArgNum, CachedAnalysis.TLI);
    //   return CachedAnalysis.MSSA.getWalker()->getClobberingMemoryAccess(BeforeCB, Loc);
    // }

	public:
		LambdaArgVisitor(CallBase* LambdaCB_, Module& M,
		                 DenseMap<Function*, std::unique_ptr<FnMemCtx>>& Cache_)
		    : LambdaCB(LambdaCB_), DL(M.getDataLayout()), FunctionAnalysisCache(Cache_), Offset(0) {
      auto* ClosurePtr = LambdaCB->getArgOperand(0);
      // auto* Clobber = findClobberingWriteBeforeCB(LambdaCB, 0);
      // if (auto *MD = dyn_cast<MemoryDef>(Clobber);  MD->getMemoryInst()) {
      //   llvm::outs() << "STARTING ANALYSIS AT " << *MD->getMemoryInst() <<"\n";
      //   PointerMayBeCapturedBefore()
      //   WorkList.push_back(MD->getMemoryInst());
      // }
      WorkList.push_back(ClosurePtr);
		}
	void visitStoreInst (StoreInst &SI) { WorkList.push_back(SI.getValueOperand()); }
  void visitCallBase (CallBase &) {}
	void visitLoadInst (LoadInst &LI) {
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

auto back() { return WorkList.back(); }
void popBack() { WorkList.pop_back(); }
bool seen(Value* Val) { return Seen.contains(Val); }
void markAsSeen(Value* Val) { Seen.insert(Val); }
bool empty() { return WorkList.empty(); }
bool success() { return AnalysisSuccess; }
bool failed() { return AnalysisFailed; }
auto getKernelArgAndOffset() { return std::make_pair(KernelArg, Offset); }

void visitGetElementPtrInst (GetElementPtrInst &GEP) {
  int64_t GEPOffset = 0;
  GetPointerBaseWithConstantOffset(&GEP, GEPOffset, DL);
  Offset+=GEPOffset;
  // llvm::outs()<<"COMPUTED OFFSET " << GEPOffset << "\n";
  WorkList.push_back(GEP.getPointerOperand());
}

// todo: these three methods need to be changed to find a dominating store
void visitAllocaInst (AllocaInst &Alloca) {
  for (auto* User : Alloca.users())
    if (!Seen.contains(User))
      WorkList.push_back(User);
}

void visitBitCastInst (BitCastInst &BC) {
  for (auto* User : BC.users())
    if (!Seen.contains(User))
      WorkList.push_back(User);
}

void visitAddrSpaceCastInst (AddrSpaceCastInst &ASC) {
  for (auto* User : ASC.users())
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
  Value *DstBase = GetPointerBaseWithConstantOffset(MT->getRawDest(),   DstOff, DL);
  Value *SrcBase = GetPointerBaseWithConstantOffset(MT->getRawSource(), SrcOff, DL);
  if (!DstBase || !SrcBase) {
    AnalysisFailed = true;
    return;
  }

  // Optional safety: only valid if the tracked byte lies within the copied region.
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

void visitIntrinsicInst (IntrinsicInst &II) {
  AnalysisFailed = true;
  return;
}

void visitArgument(Argument &A) {
  // llvm::outs() << "VISITING ARG " << A;
  Function* F = A.getParent();
  // llvm::outs() << "PARENT = " << *F;
  auto ArgNum = A.getArgNo();
  // termination case:  we have reached the parent calling kernel
  // todo: we could just pass in the kernel pointer here and check equality
  if (F->getCallingConv() == CallingConv::AMDGPU_KERNEL ||
      F->getCallingConv() == CallingConv::PTX_Kernel) {
    AnalysisSuccess = true;
    KernelArg = ArgNum;
    return;
  }
  if (!FunctionAnalysisCache.contains(F))
    FunctionAnalysisCache[F] = std::make_unique<FnMemCtx>(*F);

  for (User* U : F->users()) {
    auto* CB = dyn_cast<CallBase>(U);
    if (!CB)
      continue;
    // llvm::outs() << "    USER = " << *U << "\n";
    WorkList.push_back(CB->getArgOperand(ArgNum));
    // auto *Clobber = findClobberingWriteBeforeCB(CB, ArgNum);
    // if (auto *MD = dyn_cast<MemoryDef>(Clobber);  MD->getMemoryInst()) {
    //   Instruction *Writer = MD->getMemoryInst();
    //   // llvm::outs() << "    FOUND ACCESS " << *Writer << " DOMINATING " << A <<"\n";
    //   WorkList.push_back(Writer);
    // }
  }
}

void visitInstruction (Instruction&) {
  AnalysisFailed = true;
  return;
}

// todo add tests for the below two methods
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
    Value *Base = GetPointerBaseWithConstantOffset(Inc, IncOff, DL);
    Base = Base->stripPointerCasts();

    if (First) {
      CommonBase = Base;
      CommonIncOff = IncOff;
      First = false;
      continue;
    }

    if (Base != CommonBase || IncOff != CommonIncOff) {
      AnalysisFailed = true; // would need path-sensitive offsets to proceed
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
    AnalysisFailed = true; // would need path-sensitive offsets to proceed
    return;
  }

  WorkList.push_back(TBase);
  Offset += TOff; // select == TBase + TOff
}

};

inline bool analyzeLambdaUses(llvm::Module &M,
 DenseMap<CallBase*, std::pair<uint32_t, int64_t>> &CallBaseToArgOffset,
	  const SmallVector<CallBase*> &CBToAnalyze,
    DenseMap<Function*, std::unique_ptr<FnMemCtx>>& FunctionAnalysisCache) {
	  for (auto* FunctorCB : CBToAnalyze) {
	    LambdaArgVisitor Visitor (FunctorCB, M, FunctionAnalysisCache);
	    while (!Visitor.empty() && !Visitor.success() && !Visitor.failed()) {
	      auto* V = (Visitor.back());
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
		    CallBaseToArgOffset[FunctorCB] = Visitor.getKernelArgAndOffset();
		  }
		  return true;
		}
}
#endif
