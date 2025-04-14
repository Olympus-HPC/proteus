#include "proteus/NonAffineSCEVs.hpp"
#include "proteus/DetectScops.hpp"
#include "polly/CodeGen/CodeGeneration.h"
#include "polly/LinkAllPasses.h"
#include "polly/Options.h"
#include "polly/ScopDetection.h"
#include "polly/ScopDetectionDiagnostic.h"
#include "polly/Support/SCEVValidator.h"
#include "polly/Support/ScopLocation.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/Delinearization.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/RegionIterator.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/IntrinsicInst.h"

//#include "proteus/log.h"
#include <llvm/Analysis/AliasAnalysis.h>
#include <stack>

using namespace llvm;
using namespace proteus;

#define DEBUG_TYPE "jitpass"
#ifdef PROTEUS_ENABLE_DEBUG
#define DEBUG(x) x
#else
#define DEBUG(x)
#endif

namespace {
//REGISTER_LOG(console, DEBUG_TYPE);
} // namespace

namespace proteus {
bool PollyTrackFailures = false;
bool PollyProcessUnprofitable;
StringRef PollySkipFnAttr = "polly.skip.fn";
} // namespace proteus

extern llvm::cl::OptionCategory ProteusCategory;
extern llvm::cl::OptionCategory PolyJitRuntime;
extern llvm::cl::OptionCategory PolyJitCompiletime;

// This option is set to a very high value, as analyzing such loops increases
// compile time on several cases. For experiments that enable this option,
// a value of around 40 has been working to avoid run-time regressions with
// Polly while still exposing interesting optimization opportunities.
static cl::opt<int> ProfitabilityMinPerLoopInstructions(
    "proteus-detect-profitability-min-per-loop-insts",
    cl::desc("The minimal number of per-loop instructions before a single loop "
             "region is considered profitable"),
    cl::Hidden, cl::ValueRequired, cl::init(100000000), cl::cat(ProteusCategory));

bool PollyProcessUnprofitable;
static cl::opt<bool, true> XPollyProcessUnprofitable(
    "proteus-process-unprofitable",
    cl::desc(
        "Process scops that are unlikely to benefit from Polly optimizations."),
    cl::location(proteus::PollyProcessUnprofitable), cl::init(false),
    cl::ZeroOrMore, cl::cat(ProteusCategory));

static cl::opt<std::string> OnlyFunction(
    "proteus-only-func",
    cl::desc("Only run on functions that contain a certain string"),
    cl::value_desc("string"), cl::ValueRequired, cl::init(""),
    cl::cat(ProteusCategory));

static cl::opt<std::string> OnlyRegion(
    "proteus-only-region",
    cl::desc("Only run on certain regions (The provided identifier must "
             "appear in the name of the region's entry block"),
    cl::value_desc("identifier"), cl::ValueRequired, cl::init(""),
    cl::cat(ProteusCategory));

static cl::opt<bool>
    ReportLevel("proteus-report",
                cl::desc("Print information about the activities of Polly"),
                cl::init(false), cl::ZeroOrMore, cl::cat(ProteusCategory));

static cl::opt<bool> AllowDifferentTypes(
    "proteus-allow-differing-element-types",
    cl::desc("Allow different element types for array accesses"), cl::Hidden,
    cl::init(true), cl::ZeroOrMore, cl::cat(ProteusCategory));

static cl::opt<bool>
    AllowModrefCall("proteus-allow-modref-calls",
                    cl::desc("Allow functions with known modref behavior"),
                    cl::Hidden, cl::init(false), cl::ZeroOrMore,
                    cl::cat(ProteusCategory));

static cl::opt<bool, true>
    TrackFailures("proteus-detect-track-failures",
                  cl::desc("Track failure strings in detecting scop regions"),
                  cl::location(proteus::PollyTrackFailures), cl::Hidden,
                  cl::ZeroOrMore, cl::init(true), cl::cat(ProteusCategory));

static cl::opt<bool> KeepGoing("proteus-detect-keep-going",
                               cl::desc("Do not fail on the first error."),
                               cl::Hidden, cl::ZeroOrMore, cl::init(false),
                               cl::cat(ProteusCategory));

static cl::opt<bool>
    VerifyScops("proteus-detect-verify",
                cl::desc("Verify the detected SCoPs after each transformation"),
                cl::Hidden, cl::init(false), cl::ZeroOrMore,
                cl::cat(ProteusCategory));

/// @brief The minimal trip count under which loops are considered unprofitable.
static const unsigned MinLoopTripCount = 8;

//===----------------------------------------------------------------------===//
// Statistics.

STATISTIC(ValidRegion,
          "Number of regions that are a valid Scop at compile time.");
STATISTIC(JitRegion, "Number of jitable SCoPs");

namespace proteus {
class DiagnosticScopFound : public DiagnosticInfo {
private:
  static int PluginDiagnosticKind;

  Function &F;
  std::string FileName;
  unsigned EntryLine, ExitLine;

public:
  DiagnosticScopFound(Function &F, std::string FileName, unsigned EntryLine,
                      unsigned ExitLine)
      : DiagnosticInfo(PluginDiagnosticKind, DS_Note), F(F), FileName(FileName),
        EntryLine(EntryLine), ExitLine(ExitLine) {}

  virtual void print(DiagnosticPrinter &DP) const;

  static bool classof(const DiagnosticInfo *DI) {
    return DI->getKind() == PluginDiagnosticKind;
  }
};

int DiagnosticScopFound::PluginDiagnosticKind =
    getNextAvailablePluginDiagnosticKind();

void DiagnosticScopFound::print(DiagnosticPrinter &DP) const {
  DP << "Polly detected an optimizable loop region (scop) in function '" << F
     << "'\n";

  if (FileName.empty()) {
    DP << "Scop location is unknown. Compile with debug info "
          "(-g) to get more precise information. ";
    return;
  }

  DP << FileName << ":" << EntryLine << ": Start of scop\n";
  DP << FileName << ":" << ExitLine << ": End of scop";
}
} // namespace proteus

//===----------------------------------------------------------------------===//
// JITScopDetection.

JITScopDetection::JITScopDetection() : FunctionPass(ID) {}

template <class RR, typename... Args>
inline bool JITScopDetection::invalid(DetectionContext &Context, bool Assert,
                                      Args &&... Arguments) const {

  RI->printStyle = Region::PrintStyle::PrintBB;
  if (!Context.Verifying) {
    polly::RejectLog &Log = Context.Log;
    std::shared_ptr<RR> RejectReason = std::make_shared<RR>(Arguments...);

    if (PollyTrackFailures)
      Log.report(RejectReason);

    DEBUG({
      console->debug("\nR: {:s}\n"
                     "    {:s}",
                     Context.CurRegion.getNameStr(),
                     RejectReason->getMessage());
    });
  } else {
    assert(!Assert && "Verification of detected scop failed");
  }

  return false;
}

bool JITScopDetection::isMaxRegionInScop(const Region &R, bool Verify) const {
  if (!ValidRegions.count(&R))
    return false;

  if (Verify) {
    DetectionContextMap.erase(polly::getBBPairForRegion(&R));
    const auto &It = DetectionContextMap.insert(std::make_pair(
        polly::getBBPairForRegion(&R),
        DetectionContext(const_cast<Region &>(R), *BAA, false /*verifying*/)));
    DetectionContext &Context = It.first->second;
    return isValidRegion(Context);
  }

  return true;
}

std::string JITScopDetection::regionIsInvalidBecause(const Region *R) const {
  // Get the first error we found. Even in keep-going mode, this is the first
  // reason that caused the candidate to be rejected.
  auto *Log = lookupRejectionLog(R);

  // This can happen when we marked a region invalid, but didn't track
  // an error for it.
  if (!Log || !Log->hasErrors())
    return "";

  polly::RejectReasonPtr RR = *Log->begin();
  return RR->getMessage();
}

bool JITScopDetection::addOverApproximatedRegion(
    Region *AR, DetectionContext &Context) const {

  // If we already know about Ar we can exit.
  if (!Context.NonAffineSubRegionSet.insert(AR))
    return true;

  // All loops in the region have to be overapproximated too if there
  // are accesses that depend on the iteration count.
  for (BasicBlock *BB : AR->blocks()) {
    Loop *L = LI->getLoopFor(BB);
    if (AR->contains(L))
      Context.BoxedLoopsSet.insert(L);
  }

  return Context.BoxedLoopsSet.empty();
}

bool JITScopDetection::onlyValidRequiredInvariantLoads(
    polly::InvariantLoadsSetTy &RequiredILS, DetectionContext &Context) const {
  Region &CurRegion = Context.CurRegion;

  if (!polly::PollyInvariantLoadHoisting && !RequiredILS.empty())
    return false;

  for (LoadInst *Load : RequiredILS)
    if (!polly::isHoistableLoad(Load, CurRegion, *LI, *SE, *DT, RequiredILS))
      return false;

  Context.RequiredILS.insert(RequiredILS.begin(), RequiredILS.end());

  return true;
}

bool JITScopDetection::isAffine(const SCEV *S, Loop *Scope,
                                DetectionContext &Context,
                                Value *BaseAddress) const {

  polly::InvariantLoadsSetTy AccessILS;
  bool IsAffine =
      polly::isAffineExpr(&Context.CurRegion, Scope, S, *SE, &AccessILS);

  if (!IsAffine) {
    bool IsValidNonAffine =
        isNonAffineExpr(&Context.CurRegion, Scope, S, *SE, &AccessILS);
    if (!IsValidNonAffine)
      return false;

    for (auto P : getParamsInNonAffineExpr(&Context.CurRegion, Scope, S, *SE))
      Context.RequiredParams.push_back(P);
    Context.requiresJIT = true;
  } else {
    for (auto P :
         polly::getParamsInAffineExpr(&Context.CurRegion, Scope, S, *SE))
      Context.RequiredParams.push_back(P);
  }

  if (!onlyValidRequiredInvariantLoads(AccessILS, Context))
    return false;

  return true;
}

bool JITScopDetection::isValidSwitch(BasicBlock &BB, SwitchInst *SI,
                                     Value *Condition, bool IsLoopBranch,
                                     DetectionContext &Context) const {
  Loop *L = LI->getLoopFor(&BB);
  const SCEV *ConditionSCEV = SE->getSCEVAtScope(Condition, L);

  if (isAffine(ConditionSCEV, L, Context))
    return true;

  if (!IsLoopBranch &&
      addOverApproximatedRegion(RI->getRegionFor(&BB), Context))
    return true;

  if (IsLoopBranch)
    return false;

  return invalid<polly::ReportNonAffBranch>(Context, /*Assert=*/true, &BB,
                                            ConditionSCEV, ConditionSCEV, SI);
}

bool JITScopDetection::isValidBranch(BasicBlock &BB, BranchInst *BI,
                                     Value *Condition, bool IsLoopBranch,
                                     DetectionContext &Context) const {

  if (BinaryOperator *BinOp = dyn_cast<BinaryOperator>(Condition)) {
    auto Opcode = BinOp->getOpcode();
    if (Opcode == Instruction::And || Opcode == Instruction::Or) {
      Value *Op0 = BinOp->getOperand(0);
      Value *Op1 = BinOp->getOperand(1);
      return isValidBranch(BB, BI, Op0, IsLoopBranch, Context) &&
             isValidBranch(BB, BI, Op1, IsLoopBranch, Context);
    }
  }

  // Non constant conditions of branches need to be ICmpInst.
  if (!isa<ICmpInst>(Condition)) {
    if (!IsLoopBranch &&
        addOverApproximatedRegion(RI->getRegionFor(&BB), Context))
      return true;
    return invalid<polly::ReportInvalidCond>(Context, /*Assert=*/true, BI, &BB);
  }

  ICmpInst *ICmp = cast<ICmpInst>(Condition);

  // Are both operands of the ICmp affine?
  if (isa<UndefValue>(ICmp->getOperand(0)) ||
      isa<UndefValue>(ICmp->getOperand(1)))
    return invalid<polly::ReportUndefOperand>(Context, /*Assert=*/true, &BB,
                                              ICmp);

  Loop *L = LI->getLoopFor(ICmp->getParent());
  const SCEV *LHS = SE->getSCEVAtScope(ICmp->getOperand(0), L);
  const SCEV *RHS = SE->getSCEVAtScope(ICmp->getOperand(1), L);

  if (isAffine(LHS, L, Context) && isAffine(RHS, L, Context))
    return true;

  if (!IsLoopBranch &&
      addOverApproximatedRegion(RI->getRegionFor(&BB), Context))
    return true;

  if (IsLoopBranch)
    return false;

  return invalid<polly::ReportNonAffBranch>(Context, /*Assert=*/true, &BB, LHS,
                                            RHS, ICmp);
}

bool JITScopDetection::isValidCFG(BasicBlock &BB, bool IsLoopBranch,
                                  bool AllowUnreachable,
                                  DetectionContext &Context) const {
  Region &CurRegion = Context.CurRegion;

  Instruction *TI = BB.getTerminator();

  if (AllowUnreachable && isa<UnreachableInst>(TI))
    return true;

  // Return instructions are only valid if the region is the top level region.
  if (isa<ReturnInst>(TI) && !CurRegion.getExit() && TI->getNumOperands() == 0)
    return true;

  Value *Condition = polly::getConditionFromTerminator(TI);

  if (!Condition)
    return invalid<polly::ReportInvalidTerminator>(Context, /*Assert=*/true,
                                                   &BB);

  // UndefValue is not allowed as condition.
  if (isa<UndefValue>(Condition))
    return invalid<polly::ReportUndefCond>(Context, /*Assert=*/true, TI, &BB);

  // Constant integer conditions are always affine.
  if (isa<ConstantInt>(Condition))
    return true;

  if (BranchInst *BI = dyn_cast<BranchInst>(TI))
    return isValidBranch(BB, BI, Condition, IsLoopBranch, Context);

  SwitchInst *SI = dyn_cast<SwitchInst>(TI);
  assert(SI && "Terminator was neither branch nor switch");

  return isValidSwitch(BB, SI, Condition, IsLoopBranch, Context);
}

bool JITScopDetection::isValidCallInst(CallInst &CI,
                                       DetectionContext &Context) const {
  if (CI.doesNotReturn())
    return false;

  if (CI.doesNotAccessMemory())
    return true;

  if (auto *II = dyn_cast<IntrinsicInst>(&CI))
    if (isValidIntrinsicInst(*II, Context))
      return true;

  Function *CalledFunction = CI.getCalledFunction();

  // Indirect calls are not supported.
  if (CalledFunction == 0)
    return false;

  if (AllowModrefCall) {
    const auto& MemEffects = AA->getMemoryEffects(CalledFunction);
    if (MemEffects.onlyAccessesArgPointees()) {
      //for (auto* Arg = CI.arg_begin(); Arg != CI.arg_end(); ++Arg) {
      for (const auto &Arg : CI.args()) {
        if (!Arg->getType()->isPointerTy())
          continue;

        // Bail if a pointer argument has a base address not known to
        // ScalarEvolution. Note that a zero pointer is acceptable.
        auto *ArgSCEV = SE->getSCEVAtScope(Arg, LI->getLoopFor(CI.getParent()));
        if (ArgSCEV->isZero())
          continue;

        auto *BP = dyn_cast<SCEVUnknown>(SE->getPointerBase(ArgSCEV));
        if (!BP)
          return false;

        // Implicitly disable delinearization since we have an unknown
        // accesses with an unknown access function.
        Context.HasUnknownAccess = true;
      }

      Context.AST.add(&CI);
      return true;
    } else if (MemEffects.onlyAccessesInaccessibleMem() || 
              MemEffects.onlyAccessesInaccessibleOrArgMem() ||
              MemEffects.doesNotAccessMemory()) {
      return false;
    } else if (MemEffects.doesNotAccessMemory() || MemEffects.onlyReadsMemory()) {
      // Implicitly disable delinearization since we have an unknown
      // accesses with an unknown access function.
      Context.HasUnknownAccess = true;
      Context.AST.add(&CI);
      return true;
    }
  }

  return false;
}

bool JITScopDetection::isValidIntrinsicInst(IntrinsicInst &II,
                                            DetectionContext &Context) const {
  if (polly::isIgnoredIntrinsic(&II))
    return true;

  // The closest loop surrounding the call instruction.
  Loop *L = LI->getLoopFor(II.getParent());

  // The access function and base pointer for memory intrinsics.
  const SCEV *AF;
  const SCEVUnknown *BP;

  switch (II.getIntrinsicID()) {
  // Memory intrinsics that can be represented are supported.
  case llvm::Intrinsic::memmove:
  case llvm::Intrinsic::memcpy:
    AF = SE->getSCEVAtScope(cast<MemTransferInst>(II).getSource(), L);
    if (!AF->isZero()) {
      BP = dyn_cast<SCEVUnknown>(SE->getPointerBase(AF));
      // Bail if the source pointer is not valid.
      if (!isValidAccess(&II, AF, BP, Context))
        return false;
    }
  // Fall through
  case llvm::Intrinsic::memset:
    AF = SE->getSCEVAtScope(cast<MemIntrinsic>(II).getDest(), L);
    if (!AF->isZero()) {
      BP = dyn_cast<SCEVUnknown>(SE->getPointerBase(AF));
      // Bail if the destination pointer is not valid.
      if (!isValidAccess(&II, AF, BP, Context))
        return false;
    }

    // Bail if the length is not affine.
    if (!isAffine(SE->getSCEVAtScope(cast<MemIntrinsic>(II).getLength(), L), L,
                  Context))
      return false;

    return true;
  default:
    break;
  }

  return false;
}

bool JITScopDetection::isInvariant(const Value &Val, const Region &Reg) const {
  // A reference to function argument or constant value is invariant.
  if (isa<Argument>(Val) || isa<Constant>(Val))
    return true;

  const Instruction *I = dyn_cast<Instruction>(&Val);
  if (!I)
    return false;

  if (!Reg.contains(I))
    return true;

  if (I->mayHaveSideEffects())
    return false;

  // When Val is a Phi node, it is likely not invariant. We do not check whether
  // Phi nodes are actually invariant, we assume that Phi nodes are usually not
  // invariant. Recursively checking the operators of Phi nodes would lead to
  // infinite recursion.
  if (isa<PHINode>(*I))
    return false;

  for (const Use &Operand : I->operands())
    if (!isInvariant(*Operand, Reg))
      return false;

  return true;
}

/// @brief Remove smax of smax(0, size) expressions from a SCEV expression and
/// register the '...' components.
///
/// Array access expressions as they are generated by gfortran contain smax(0,
/// size) expressions that confuse the 'normal' delinearization algorithm.
/// However, if we extract such expressions before the normal delinearization
/// takes place they can actually help to identify array size expressions in
/// fortran accesses. For the subsequently following delinearization the smax(0,
/// size) component can be replaced by just 'size'. This is correct as we will
/// always add and verify the assumption that for all subscript expressions
/// 'exp' the inequality 0 <= exp < size holds. Hence, we will also verify
/// that 0 <= size, which means smax(0, size) == size.
struct SCEVRemoveMax : public SCEVVisitor<SCEVRemoveMax, const SCEV *> {
public:
  static const SCEV *remove(ScalarEvolution &SE, const SCEV *Expr,
                            std::vector<const SCEV *> *Terms = nullptr) {

    SCEVRemoveMax D(SE, Terms);
    return D.visit(Expr);
  }

  SCEVRemoveMax(ScalarEvolution &SE, std::vector<const SCEV *> *Terms)
      : SE(SE), Terms(Terms) {}

  const SCEV *visitTruncateExpr(const SCEVTruncateExpr *Expr) { return Expr; }

  const SCEV *visitZeroExtendExpr(const SCEVZeroExtendExpr *Expr) {
    return Expr;
  }

  const SCEV *visitSignExtendExpr(const SCEVSignExtendExpr *Expr) {
    return SE.getSignExtendExpr(visit(Expr->getOperand()), Expr->getType());
  }

  const SCEV *visitUDivExpr(const SCEVUDivExpr *Expr) { return Expr; }

  const SCEV *visitSMaxExpr(const SCEVSMaxExpr *Expr) {
    if ((Expr->getNumOperands() == 2) && Expr->getOperand(0)->isZero()) {
      auto Res = visit(Expr->getOperand(1));
      if (Terms)
        (*Terms).push_back(Res);
      return Res;
    }

    return Expr;
  }

  const SCEV *visitUMaxExpr(const SCEVUMaxExpr *Expr) { return Expr; }

  const SCEV *visitUnknown(const SCEVUnknown *Expr) { return Expr; }

  const SCEV *visitCouldNotCompute(const SCEVCouldNotCompute *Expr) {
    return Expr;
  }

  const SCEV *visitConstant(const SCEVConstant *Expr) { return Expr; }

  const SCEV *visitAddRecExpr(const SCEVAddRecExpr *Expr) {
    SmallVector<const SCEV *, 5> NewOps;
    for (const SCEV *Op : Expr->operands())
      NewOps.push_back(visit(Op));

    return SE.getAddRecExpr(NewOps, Expr->getLoop(), Expr->getNoWrapFlags());
  }

  const SCEV *visitAddExpr(const SCEVAddExpr *Expr) {
    SmallVector<const SCEV *, 5> NewOps;
    for (const SCEV *Op : Expr->operands())
      NewOps.push_back(visit(Op));

    return SE.getAddExpr(NewOps);
  }

  const SCEV *visitMulExpr(const SCEVMulExpr *Expr) {
    SmallVector<const SCEV *, 5> NewOps;
    for (const SCEV *Op : Expr->operands())
      NewOps.push_back(visit(Op));

    return SE.getMulExpr(NewOps);
  }

private:
  ScalarEvolution &SE;
  std::vector<const SCEV *> *Terms;
};

SmallVector<const SCEV *, 4> JITScopDetection::getDelinearizationTerms(
    DetectionContext &Context, const SCEVUnknown *BasePointer) const {
  SmallVector<const SCEV *, 4> Terms;
  for (const auto &Pair : Context.Accesses[BasePointer]) {
    std::vector<const SCEV *> MaxTerms;
    SCEVRemoveMax::remove(*SE, Pair.second, &MaxTerms);
    if (MaxTerms.size() > 0) {
      Terms.insert(Terms.begin(), MaxTerms.begin(), MaxTerms.end());
      continue;
    }
    // In case the outermost expression is a plain add, we check if any of its
    // terms has the form 4 * %inst * %param * %param ..., aka a term that
    // contains a product between a parameter and an instruction that is
    // inside the scop. Such instructions, if allowed at all, are instructions
    // SCEV can not represent, but Polly is still looking through. As a
    // result, these instructions can depend on induction variables and are
    // most likely no array sizes. However, terms that are multiplied with
    // them are likely candidates for array sizes.
    if (auto *AF = dyn_cast<SCEVAddExpr>(Pair.second)) {
      for (auto Op : AF->operands()) {
        if (auto *AF2 = dyn_cast<SCEVAddRecExpr>(Op))
          collectParametricTerms(*SE, AF2, Terms);
        if (auto *AF2 = dyn_cast<SCEVMulExpr>(Op)) {
          SmallVector<const SCEV *, 0> Operands;

          for (auto *MulOp : AF2->operands()) {
            if (auto *Const = dyn_cast<SCEVConstant>(MulOp))
              Operands.push_back(Const);
            if (auto *Unknown = dyn_cast<SCEVUnknown>(MulOp)) {
              if (auto *Inst = dyn_cast<Instruction>(Unknown->getValue())) {
                if (!Context.CurRegion.contains(Inst))
                  Operands.push_back(MulOp);

              } else {
                Operands.push_back(MulOp);
              }
            }
          }
          if (Operands.size())
            Terms.push_back(SE->getMulExpr(Operands));
        }
      }
    }
    if (Terms.empty())
      collectParametricTerms(*SE, Pair.second, Terms);
  }
  return Terms;
}

bool JITScopDetection::isValidAccess(Instruction *Inst, const SCEV *AF,
                                     const SCEVUnknown *BP,
                                     DetectionContext &Context) const {

  if (!BP)
    return invalid<polly::ReportNoBasePtr>(Context, /*Assert=*/true, Inst);

  auto *BV = BP->getValue();
  if (isa<UndefValue>(BV))
    return invalid<polly::ReportUndefBasePtr>(Context, /*Assert=*/true, Inst);

  // FIXME: Think about allowing IntToPtrInst
  if (IntToPtrInst *Inst = dyn_cast<IntToPtrInst>(BV))
    return invalid<polly::ReportIntToPtr>(Context, /*Assert=*/true, Inst);

  // Check that the base address of the access is invariant in the current
  // region.
  if (!isInvariant(*BV, Context.CurRegion))
    return invalid<polly::ReportVariantBasePtr>(Context, /*Assert=*/true, BV,
                                                Inst);

  AF = SE->getMinusSCEV(AF, BP);

  const SCEV *Size;
  if (!isa<MemIntrinsic>(Inst)) {
    Size = SE->getElementSize(Inst);
  } else {
    auto *SizeTy =
        SE->getEffectiveSCEVType(PointerType::getInt8PtrTy(SE->getContext()));
    Size = SE->getConstant(SizeTy, 8);
  }

  if (Context.ElementSize[BP]) {
    if (!AllowDifferentTypes && Context.ElementSize[BP] != Size)
      return invalid<polly::ReportDifferentArrayElementSize>(
          Context, /*Assert=*/true, Inst, BV);

    Context.ElementSize[BP] = SE->getSMinExpr(Size, Context.ElementSize[BP]);
  } else {
    Context.ElementSize[BP] = Size;
  }

  bool IsVariantInNonAffineLoop = false;
  SetVector<const Loop *> Loops;
  polly::findLoops(AF, Loops);
  for (const Loop *L : Loops)
    if (Context.BoxedLoopsSet.count(L))
      IsVariantInNonAffineLoop = true;

  auto *Scope = LI->getLoopFor(Inst->getParent());
  bool IsAffine = !IsVariantInNonAffineLoop && isAffine(AF, Scope, Context, BV);
  // Do not try to delinearize memory intrinsics and force them to be affine.
  if (isa<MemIntrinsic>(Inst) && !IsAffine) {
    return invalid<polly::ReportNonAffineAccess>(Context, /*Assert=*/true, AF,
                                                 Inst, BV);
  } else if (!IsAffine) {
    return invalid<polly::ReportNonAffineAccess>(Context, /*Assert=*/true, AF,
                                                 Inst, BV);
  }

  // Check if the base pointer of the memory access does alias with
  // any other pointer. This cannot be handled at the moment.
  AAMDNodes AATags = Inst->getAAMetadata();
  AliasSet &AS = Context.AST.getAliasSetFor(
      MemoryLocation(BP->getValue(), MemoryLocation::UnknownSize, AATags));

  if (!AS.isMustAlias()) {
    bool CanBuildRunTimeCheck = true;
    // The run-time alias check places code that involves the base pointer at
    // the beginning of the SCoP. This breaks if the base pointer is defined
    // inside the scop. Hence, we can only create a run-time check if we are
    // sure the base pointer is not an instruction defined inside the scop.
    // However, we can ignore loads that will be hoisted.

    // In PolyJIT's version of the RTC, we only check if it is possible to
    // generate a RTC. The check itself will be created at run-time to enable
    // further optimization.
    for (const auto &Ptr : AS) {
      Instruction *Inst = dyn_cast<Instruction>(Ptr.getValue());
      if (Inst && Context.CurRegion.contains(Inst)) {
        auto *Load = dyn_cast<LoadInst>(Inst);
        Context.RequiredILS.insert(Load);
        if (Load && polly::isHoistableLoad(Load, Context.CurRegion, *LI, *SE, *DT, getRequiredInvariantLoads(&Context.CurRegion))) {
          continue;
        }

        CanBuildRunTimeCheck = false;
        break;
      }
    }

    if (CanBuildRunTimeCheck)
      return true;
    return invalid<polly::ReportAlias>(Context, /*Assert=*/true, Inst, AS);
  }

  return true;
}

bool JITScopDetection::isValidMemoryAccess(polly::MemAccInst Inst,
                                           DetectionContext &Context) const {
  Value *Ptr = Inst.getPointerOperand();
  Loop *L = LI->getLoopFor(Inst->getParent());
  const SCEV *AccessFunction = SE->getSCEVAtScope(Ptr, L);
  const SCEVUnknown *BasePointer;

  BasePointer = dyn_cast<SCEVUnknown>(SE->getPointerBase(AccessFunction));

  return isValidAccess(Inst, AccessFunction, BasePointer, Context);
}

bool JITScopDetection::isValidInstruction(Instruction &Inst,
                                          DetectionContext &Context) const {
  for (auto &Op : Inst.operands()) {
    auto *OpInst = dyn_cast<Instruction>(&Op);

    if (!OpInst)
      continue;

    if (polly::isErrorBlock(*OpInst->getParent(), Context.CurRegion, *LI, *DT))
      return false;
  }

  if (isa<LandingPadInst>(&Inst) || isa<ResumeInst>(&Inst))
    return false;

  // We only check the call instruction but not invoke instruction.
  if (CallInst *CI = dyn_cast<CallInst>(&Inst)) {
    if (isValidCallInst(*CI, Context))
      return true;

    return invalid<polly::ReportFuncCall>(Context, /*Assert=*/true, &Inst);
  }

  if (!Inst.mayWriteToMemory() && !Inst.mayReadFromMemory()) {
    if (!isa<AllocaInst>(Inst))
      return true;

    return invalid<polly::ReportAlloca>(Context, /*Assert=*/true, &Inst);
  }

  // Check the access function.
  if (auto MemInst = polly::MemAccInst::dyn_cast(Inst)) {
    Context.hasStores |= isa<StoreInst>(MemInst);
    Context.hasLoads |= isa<LoadInst>(MemInst);
    if (!MemInst.isSimple())
      return invalid<polly::ReportNonSimpleMemoryAccess>(
          Context, /*Assert=*/true, &Inst);

    return isValidMemoryAccess(MemInst, Context);
  }

  // We do not know this instruction, therefore we assume it is invalid.
  return invalid<polly::ReportUnknownInst>(Context, /*Assert=*/true, &Inst);
}

bool JITScopDetection::canUseISLTripCount(Loop *L,
                                          DetectionContext &Context) const {
  // Ensure the loop has valid exiting blocks as well as latches, otherwise we
  // need to overapproximate it as a boxed loop.
  SmallVector<BasicBlock *, 4> LoopControlBlocks;
  L->getExitingBlocks(LoopControlBlocks);
  // Loops without exiting blocks cannot be handled by the schedule generation
  // as it depends on a region covering that is not given.
  if (LoopControlBlocks.empty())
    return false;

  L->getLoopLatches(LoopControlBlocks);
  for (BasicBlock *ControlBB : LoopControlBlocks) {
    if (!isValidCFG(*ControlBB, true, false, Context))
      return false;
  }

  // We can use ISL to compute the trip count of L.
  return true;
}

bool JITScopDetection::isValidLoop(Loop *L, DetectionContext &Context) const {
  Region *R = RI->getRegionFor(L->getHeader());
  while (R != &Context.CurRegion && !R->contains(L))
    R = R->getParent();

  const SCEV *LoopCount = SE->getBackedgeTakenCount(L);
  if (proteus::isNonAffineExpr(R, L, LoopCount, *SE, nullptr)) {
    for (auto P : getParamsInNonAffineExpr(R, L, LoopCount, *SE))
      Context.RequiredParams.push_back(P);
    Context.requiresJIT = true;
    return true;
  }

  if (canUseISLTripCount(L, Context))
    return true;

  if (addOverApproximatedRegion(R, Context))
    return true;

  return invalid<polly::ReportLoopBound>(Context, /*Assert=*/true, L,
                                         LoopCount);
}

/// @brief Return the number of loops in @p L (incl. @p L) that have a trip
///        count that is not known to be less than MIN_LOOP_TRIP_COUNT.
static int countBeneficialSubLoops(Loop *L, ScalarEvolution &SE) {
  auto *TripCount = SE.getBackedgeTakenCount(L);

  int Count = 1;
  if (auto *TripCountC = dyn_cast<SCEVConstant>(TripCount))
    if (TripCountC->getType()->getScalarSizeInBits() <= 64)
      if (TripCountC->getValue()->getZExtValue() < MinLoopTripCount)
        Count -= 1;

  for (auto &SubLoop : *L)
    Count += countBeneficialSubLoops(SubLoop, SE);

  return Count;
}

int JITScopDetection::countBeneficialLoops(Region *R) const {
  int LoopNum = 0;

  auto L = LI->getLoopFor(R->getEntry());
  L = L ? R->outermostLoopInRegion(L) : nullptr;
  L = L ? L->getParentLoop() : nullptr;

  auto SubLoops =
      L ? L->getSubLoopsVector() : std::vector<Loop *>(LI->begin(), LI->end());

  for (auto &SubLoop : SubLoops)
    if (R->contains(SubLoop))
      LoopNum += countBeneficialSubLoops(SubLoop, *SE);

  return LoopNum;
}

Region *JITScopDetection::expandRegion(Region &R) {
  // Initial no valid region was found (greater than R)
  std::unique_ptr<Region> LastValidRegion;
  auto ExpandedRegion = std::unique_ptr<Region>(R.getExpandedRegion());

  DEBUG(dbgs() << "\tExpanding " << R.getNameStr() << "\n");

  while (ExpandedRegion) {
    const auto &It = DetectionContextMap.insert(std::make_pair(
        polly::getBBPairForRegion(ExpandedRegion.get()),
        DetectionContext(*ExpandedRegion, BAA, false /*verifying*/)));
    DetectionContext &Context = It.first->second;
    DEBUG(dbgs() << "\t\tTrying " << ExpandedRegion->getNameStr() << "\n");
    // Only expand when we did not collect errors.

    if (!Context.Log.hasErrors()) {
      // If the exit is valid check all blocks
      //  - if true, a valid region was found => store it + keep expanding
      //  - if false, .tbd. => stop  (should this really end the loop?)
      if (!allBlocksValid(Context) || Context.Log.hasErrors()) {
        removeCachedResults(*ExpandedRegion);
        break;
      }

      // Store this region, because it is the greatest valid (encountered so
      // far).
      removeCachedResults(*LastValidRegion);
      LastValidRegion = std::move(ExpandedRegion);

      // Create and test the next greater region (if any)
      ExpandedRegion =
          std::unique_ptr<Region>(LastValidRegion->getExpandedRegion());

    } else {
      // Create and test the next greater region (if any)
      removeCachedResults(*ExpandedRegion);
      ExpandedRegion =
          std::unique_ptr<Region>(ExpandedRegion->getExpandedRegion());
    }
  }

  DEBUG({
    if (LastValidRegion)
      dbgs() << "\tto " << LastValidRegion->getNameStr() << "\n";
    else
      dbgs() << "\tExpanding " << R.getNameStr() << " failed\n";
  });

  return LastValidRegion.release();
}
static bool regionWithoutLoops(Region &R, LoopInfo *LI) {
  for (const BasicBlock *BB : R.blocks())
    if (R.contains(LI->getLoopFor(BB)))
      return false;

  return true;
}

unsigned JITScopDetection::removeCachedResultsRecursively(const Region &R,
                                                          RegionSet &Cache) {
  unsigned Count = 0;
  for (auto &SubRegion : R) {
    if (Cache.count(SubRegion.get())) {
      removeCachedResults(*SubRegion.get(), Cache);
      ++Count;
    } else
      Count += removeCachedResultsRecursively(*SubRegion, Cache);
  }
  return Count;
}

void JITScopDetection::removeCachedResults(const Region &R) {
  removeCachedResults(R, ValidRegions);
  removeCachedResults(R, ValidRuntimeRegions);
}

void JITScopDetection::removeCachedResults(const Region &R, RegionSet &Cache) {
  Cache.remove(&R);
}

void JITScopDetection::findScops(Region &R) {
  const auto &It = DetectionContextMap.insert(
      std::make_pair(polly::getBBPairForRegion(&R),
                     DetectionContext(R, BAA, false /*verifying*/)));
  DetectionContext &Context = It.first->second;

  bool RegionIsValid = false;
  if (!PollyProcessUnprofitable && regionWithoutLoops(R, LI))
    invalid<polly::ReportUnprofitable>(Context, /*Assert=*/true, &R);
  else
    RegionIsValid = isValidRegion(Context);

  bool HasErrors = !RegionIsValid || Context.Log.size() > 0;

  if (HasErrors) {
    removeCachedResults(R);
  }

  if (!HasErrors) {
    if (Context.requiresJIT)
      ValidRuntimeRegions.insert(&R);
    ValidRegions.insert(&R);

    RequiredParams[&R] = Context.RequiredParams;
    return;
  }

  for (auto &SubRegion : R)
    findScops(*SubRegion);

  // Try to expand regions.
  //
  // As the region tree normally only contains canonical regions, non canonical
  // regions that form a Scop are not found. Therefore, those non canonical
  // regions are checked by expanding the canonical ones.

  std::vector<Region *> ToExpand;

  for (auto &SubRegion : R)
    ToExpand.push_back(SubRegion.get());

  for (Region *CurrentRegion : ToExpand) {
    // Skip invalid regions. Regions may become invalid, if they are element of
    // an already expanded region.
    if (!ValidRegions.count(CurrentRegion))
      continue;

    // Skip regions that had errors.
    bool HadErrors = lookupRejectionLog(CurrentRegion)->hasErrors();
    if (HadErrors)
      continue;

    Region *ExpandedR = expandRegion(*CurrentRegion);

    if (!ExpandedR)
      continue;

    R.addSubRegion(ExpandedR, true);

    // We could expand the region. Check, if we require the JIT for it.
    const DetectionContext *ExpandedCtx = getDetectionContext(ExpandedR);
    if (ExpandedCtx->requiresJIT)
      ValidRuntimeRegions.insert(ExpandedR);
    ValidRegions.insert(ExpandedR);
    RequiredParams[ExpandedR] = ExpandedCtx->RequiredParams;

    // Erase all (direct and indirect) children of ExpandedR from the valid
    // regions and update the number of valid regions.
    removeCachedResultsRecursively(*ExpandedR, ValidRegions);
    removeCachedResultsRecursively(*ExpandedR, ValidRuntimeRegions);
  }
}

bool JITScopDetection::allBlocksValid(DetectionContext &Context) const {
  Region &CurRegion = Context.CurRegion;

  for (const BasicBlock *BB : CurRegion.blocks()) {
    Loop *L = LI->getLoopFor(BB);
    if (L && L->getHeader() == BB && CurRegion.contains(L) &&
        (!isValidLoop(L, Context) && !KeepGoing))
      return false;
  }

  for (BasicBlock *BB : CurRegion.blocks()) {
    bool IsErrorBlock = polly::isErrorBlock(*BB, CurRegion, *LI, *DT);

    // Also check exception blocks (and possibly register them as non-affine
    // regions). Even though exception blocks are not modeled, we use them
    // to forward-propagate domain constraints during ScopInfo construction.
    if (!isValidCFG(*BB, false, IsErrorBlock, Context) && !KeepGoing)
      return false;

    if (IsErrorBlock)
      continue;

    for (BasicBlock::iterator I = BB->begin(), E = --BB->end(); I != E; ++I)
      if (!isValidInstruction(*I, Context) && !KeepGoing)
        return false;
  }

  return true;
}

bool JITScopDetection::hasSufficientCompute(DetectionContext &Context,
                                            int NumLoops) const {
  int InstCount = 0;

  for (auto *BB : Context.CurRegion.blocks())
    if (Context.CurRegion.contains(LI->getLoopFor(BB)))
      InstCount += BB->size();

  InstCount = InstCount / NumLoops;

  return InstCount >= ProfitabilityMinPerLoopInstructions;
}

bool JITScopDetection::hasPossiblyDistributableLoop(
    DetectionContext &Context) const {
  for (auto *BB : Context.CurRegion.blocks()) {
    auto *L = LI->getLoopFor(BB);
    if (!Context.CurRegion.contains(L))
      continue;
    if (Context.BoxedLoopsSet.count(L))
      continue;
    unsigned StmtsWithStoresInLoops = 0;
    for (auto *LBB : L->blocks()) {
      bool MemStore = false;
      for (auto &I : *LBB)
        MemStore |= isa<StoreInst>(&I);
      StmtsWithStoresInLoops += MemStore;
    }
    return (StmtsWithStoresInLoops > 1);
  }
  return false;
}

bool JITScopDetection::isProfitableRegion(DetectionContext &Context) const {
  Region &CurRegion = Context.CurRegion;

  if (PollyProcessUnprofitable)
    return true;

  // We can probably not do a lot on scops that only write or only read
  // data.
  if (!Context.hasStores || !Context.hasLoads)
    return invalid<polly::ReportUnprofitable>(Context, /*Assert=*/true,
                                              &CurRegion);

  int NumLoops = countBeneficialLoops(&CurRegion);

  // Scops with at least two loops may allow either loop fusion or tiling and
  // are consequently interesting to look at.
  if (NumLoops >= 1)
    return true;

  // Scops that contain a loop with a non-trivial amount of computation per
  // loop-iteration are interesting as we may be able to parallelize such
  // loops. Individual loops that have only a small amount of computation
  // per-iteration are performance-wise very fragile as any change to the
  // loop induction variables may affect performance. To not cause spurious
  // performance regressions, we do not consider such loops.
  // if (NumAffineLoops == 1 && hasSufficientCompute(Context, NumLoops))
  //  return true;

  return invalid<polly::ReportUnprofitable>(Context, /*Assert=*/true,
                                            &CurRegion);
}

bool JITScopDetection::isValidRegion(DetectionContext &Context) const {
  Region &CurRegion = Context.CurRegion;

  DEBUG(dbgs() << "Checking region: " << CurRegion.getNameStr() << "\n\t");

  if (CurRegion.isTopLevelRegion()) {
    DEBUG(dbgs() << "Top level region is invalid\n");
    return false;
  }

  if (!CurRegion.getEntry()->getName().count(OnlyRegion)) {
    DEBUG({
      dbgs() << "Region entry does not match -proteus-region-only";
      dbgs() << "\n";
    });
    return false;
  }

  // SCoP cannot contain the entry block of the function, because we need
  // to insert alloca instruction there when translate scalar to array.
  if (CurRegion.getEntry() ==
      &(CurRegion.getEntry()->getParent()->getEntryBlock()))
    return invalid<polly::ReportEntry>(Context, /*Assert=*/true,
                                       CurRegion.getEntry());

  if (!allBlocksValid(Context))
    return false;

  DebugLoc DbgLoc;
  if (!isReducibleRegion(CurRegion, DbgLoc))
    return invalid<polly::ReportIrreducibleRegion>(Context, /*Assert=*/true,
                                                   &CurRegion, DbgLoc);

  DEBUG(dbgs() << "OK\n");
  return true;
}

void JITScopDetection::markFunctionAsInvalid(Function *F) const {}

bool JITScopDetection::isValidFunction(llvm::Function &F) {
  return !F.hasFnAttribute(PollySkipFnAttr);
}

void JITScopDetection::printLocations(llvm::Function &F) {
  for (const Region *R : *this) {
    unsigned LineEntry, LineExit;
    std::string FileName;

    polly::getDebugLocation(R, LineEntry, LineExit, FileName);
    DiagnosticScopFound Diagnostic(F, FileName, LineEntry, LineExit);
    F.getContext().diagnose(Diagnostic);
  }
}

/// @brief Enum for coloring BBs in Region.
///
/// WHITE - Unvisited BB in DFS walk.
/// GREY - BBs which are currently on the DFS stack for processing.
/// BLACK - Visited and completely processed BB.
enum Color { WHITE, GREY, BLACK };

bool JITScopDetection::isReducibleRegion(Region &R, DebugLoc &DbgLoc) const {
  BasicBlock *REntry = R.getEntry();
  BasicBlock *RExit = R.getExit();
  // Map to match the color of a BasicBlock during the DFS walk.
  DenseMap<const BasicBlock *, Color> BBColorMap;
  // Stack keeping track of current BB and index of next child to be processed.
  std::stack<std::pair<BasicBlock *, unsigned>> DFSStack;

  unsigned AdjacentBlockIndex = 0;
  BasicBlock *CurrBB, *SuccBB;
  CurrBB = REntry;

  // Initialize the map for all BB with WHITE color.
  for (auto *BB : R.blocks())
    BBColorMap[BB] = WHITE;

  // Process the entry block of the Region.
  BBColorMap[CurrBB] = GREY;
  DFSStack.push(std::make_pair(CurrBB, 0));

  while (!DFSStack.empty()) {
    // Get next BB on stack to be processed.
    CurrBB = DFSStack.top().first;
    AdjacentBlockIndex = DFSStack.top().second;
    DFSStack.pop();

    // Loop to iterate over the successors of current BB.
    const Instruction *TInst = CurrBB->getTerminator();
    unsigned NSucc = TInst->getNumSuccessors();
    for (unsigned I = AdjacentBlockIndex; I < NSucc;
         ++I, ++AdjacentBlockIndex) {
      SuccBB = TInst->getSuccessor(I);

      // Checks for region exit block and self-loops in BB.
      if (SuccBB == RExit || SuccBB == CurrBB)
        continue;

      // WHITE indicates an unvisited BB in DFS walk.
      if (BBColorMap[SuccBB] == WHITE) {
        // Push the current BB and the index of the next child to be visited.
        DFSStack.push(std::make_pair(CurrBB, I + 1));
        // Push the next BB to be processed.
        DFSStack.push(std::make_pair(SuccBB, 0));
        // First time the BB is being processed.
        BBColorMap[SuccBB] = GREY;
        break;
      } else if (BBColorMap[SuccBB] == GREY) {
        // GREY indicates a loop in the control flow.
        // If the destination dominates the source, it is a natural loop
        // else, an irreducible control flow in the region is detected.
        if (!DT->dominates(SuccBB, CurrBB)) {
          // Get debug info of instruction which causes irregular control flow.
          DbgLoc = TInst->getDebugLoc();
          return false;
        }
      }
    }

    // If all children of current BB have been processed,
    // then mark that BB as fully processed.
    if (AdjacentBlockIndex == NSucc)
      BBColorMap[CurrBB] = BLACK;
  }

  return true;
}

bool JITScopDetection::runOnFunction(llvm::Function &F) {
  if (F.hasFnAttribute("polyjit-jit-candidate"))
    return false;

  LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  RI = &getAnalysis<RegionInfoPass>().getRegionInfo();
  if (!PollyProcessUnprofitable && LI->empty())
    return false;

  AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();
  BAA = &BatchAAResults(AA);
  SE = &getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  Region *TopRegion = RI->getTopLevelRegion();

  releaseMemory();

  if (OnlyFunction != "" && !F.getName().count(OnlyFunction))
    return false;

  if (!isValidFunction(F))
    return false;

  findScops(*TopRegion);

  // Prune non-profitable regions.
  for (auto &DIt : DetectionContextMap) {
    auto &DC = DIt.getSecond();
    if (DC.Log.hasErrors())
      continue;
    if (!ValidRegions.count(&DC.CurRegion))
      continue;
    if (isProfitableRegion(DC))
      continue;

    ValidRegions.remove(&DC.CurRegion);
    if (ValidRuntimeRegions.count(&DC.CurRegion))
      ValidRuntimeRegions.remove(&DC.CurRegion);
  }

  if (ReportLevel)
    printLocations(F);

  assert(ValidRegions.size() <= DetectionContextMap.size() &&
         "Cached more results than valid regions");

  JitRegion += ValidRuntimeRegions.size();
  ValidRegion += ValidRegions.size() - ValidRuntimeRegions.size();
  return false;
}

JITScopDetection::DetectionContext *
JITScopDetection::getDetectionContext(const Region *R) const {
  auto DCMIt = DetectionContextMap.find(polly::getBBPairForRegion(R));
  if (DCMIt == DetectionContextMap.end())
    return nullptr;
  return &DCMIt->second;
}

const polly::RejectLog *
JITScopDetection::lookupRejectionLog(const Region *R) const {
  const DetectionContext *DC = getDetectionContext(R);
  return DC ? &DC->Log : nullptr;
}

void proteus::JITScopDetection::verifyRegion(const Region &R) const {
  assert(isMaxRegionInScop(R) && "Expect R is a valid region.");

  DetectionContext Context(const_cast<Region &>(R), *AA, true /*verifying*/);
  isValidRegion(Context);
}

void proteus::JITScopDetection::verifyAnalysis() const {
  if (!VerifyScops)
    return;

  for (const Region *R : ValidRegions)
    verifyRegion(*R);
}

void JITScopDetection::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<LoopInfoWrapperPass>();
  AU.addRequired<ScalarEvolutionWrapperPass>();
  AU.addRequired<DominatorTreeWrapperPass>();
  // We also need AA and RegionInfo when we are verifying analysis.
  AU.addRequiredTransitive<AAResultsWrapperPass>();
  AU.addRequiredTransitive<RegionInfoPass>();
  AU.setPreservesAll();
}

void JITScopDetection::print(raw_ostream &OS, const Module *) const {
  for (const Region *R : ValidRegions)
    OS << "Valid Region for Scop: " << R->getNameStr() << '\n';

  unsigned Count = ValidRuntimeRegions.size();
  unsigned I = 0;

  OS << fmt::format("{:d} regions require runtime support:\n", Count);
  for (const Region *R : ValidRuntimeRegions) {
    if (!R || (R && !RequiredParams.count(R)))
      continue;
    const ParamVec &L = RequiredParams.at(R);
    OS.indent(2) << fmt::format("{:d} region {:s} requires {:d} params\n", I++,
                                R->getNameStr(), L.size());
    unsigned J = 0;
    for (const SCEV *S : L) {
      OS.indent(4) << fmt::format("{:d} - ", J);
      S->print(OS);
      OS << "\n";
    }

    OS << "\n";
  }
}

void JITScopDetection::releaseMemory() {
  ValidRegions.clear();
  DetectionContextMap.clear();
  ValidRuntimeRegions.clear();
  RequiredParams.clear();

  // Do not clear the invalid function set.
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getProteusJitPassPluginInfo();
}

namespace proteus {
char JITScopDetection::ID = 0;
} // namespace proteus
Pass *proteus::createScopDetectionPass() { return new JITScopDetection(); }

INITIALIZE_PASS_BEGIN(JITScopDetection, "proteus-detect-scops",
                      "proteus - Detect jitable static control parts (jSCoPs)",
                      false, false);
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass);
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass);
INITIALIZE_PASS_DEPENDENCY(RegionInfoPass);
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass);
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass);
INITIALIZE_PASS_END(JITScopDetection, "proteus-detect-scops",
                    "proteus - Detect jitable static control parts (jSCoPs)",
                    false, false)
