/* Polly's ScopDetection adapted for the use within PolyJIT
*
* Copyright © 2016 Andreas Simbürger <simbuerg@lairosiel.de>
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included
* in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
* OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
* DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
* OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
#ifndef proteus_SCOPDETECTION_H
#define proteus_SCOPDETECTION_H

#include "polly/ScopDetectionDiagnostic.h"
#include "polly/Support/ScopHelper.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Pass.h"
#include <map>
#include <memory>
#include <set>

using namespace llvm;

namespace llvm {
class LoopInfo;
class Loop;
class ScalarEvolution;
class SCEV;
class SCEVAddRecExpr;
class SCEVUnknown;
class CallInst;
class Instruction;
class Value;
class IntrinsicInst;
} // namespace llvm

namespace proteus {
typedef std::set<const SCEV *> ParamSetType;

// Description of the shape of an array.
struct ArrayShape {
  // Base pointer identifying all accesses to this array.
  const SCEVUnknown *BasePointer;

  // Sizes of each delinearized dimension.
  SmallVector<const SCEV *, 4> DelinearizedSizes;

  ArrayShape(const SCEVUnknown *B) : BasePointer(B), DelinearizedSizes() {}
};

struct MemAcc {
  const Instruction *Insn;

  // A pointer to the shape description of the array.
  std::shared_ptr<ArrayShape> Shape;

  // Subscripts computed by delinearization.
  SmallVector<const SCEV *, 4> DelinearizedSubscripts;

  MemAcc(const Instruction *I, std::shared_ptr<ArrayShape> S)
      : Insn(I), Shape(S), DelinearizedSubscripts() {}
};

typedef std::map<const Instruction *, MemAcc> MapInsnToMemAcc;
typedef std::pair<const Instruction *, const SCEV *> PairInstSCEV;
typedef std::vector<PairInstSCEV> AFs;
typedef std::map<const SCEVUnknown *, AFs> BaseToAFs;
typedef std::map<const SCEVUnknown *, const SCEV *> BaseToElSize;

/// @brief A function attribute which will cause Polly to skip the function
extern llvm::StringRef PollySkipFnAttr;

//===----------------------------------------------------------------------===//
/// @brief Pass to detect the maximal static control parts (Scops) of a
/// function.
class JITScopDetection : public FunctionPass {
public:
  typedef SetVector<const Region *> RegionSet;

  // Remember the valid regions
  RegionSet ValidRegions;

  // Remember the valid run-time regions
  RegionSet ValidRuntimeRegions;

  ///
  // @name Track required params for each detected SCoP candidate.
  // @{
  using ParamVec = std::vector<const llvm::SCEV *>;
  using RegionToParamVec = std::map<const llvm::Region *, ParamVec>;
  RegionToParamVec RequiredParams;
  ///  @}


  /// @brief Context variables for SCoP detection.
  struct DetectionContext {
    Region &CurRegion;   // The region to check.
    AliasSetTracker AST; // The AliasSetTracker to hold the alias information.
    bool Verifying;      // If we are in the verification phase?

    /// @brief Container to remember rejection reasons for this region.
    polly::RejectLog Log;

    /// @brief Map a base pointer to all access functions accessing it.
    ///
    /// This map is indexed by the base pointer. Each element of the map
    /// is a list of memory accesses that reference this base pointer.
    BaseToAFs Accesses;

    /// @brief The set of base pointers with non-affine accesses.
    ///
    /// This set contains all base pointers and the locations where they are
    /// used for memory accesses that can not be detected as affine accesses.
    SetVector<std::pair<const SCEVUnknown *, Loop *>> NonAffineAccesses;
    BaseToElSize ElementSize;

    /// @brief The region has at least one load instruction.
    bool hasLoads;

    /// @brief The region has at least one store instruction.
    bool hasStores;

    /// @brief Flag to indicate the region has at least one unknown access.
    bool HasUnknownAccess;

    /// @brief The set of non-affine subregions in the region we analyze.
    RegionSet NonAffineSubRegionSet;

    /// @brief The set of loops contained in non-affine regions.
    polly::BoxedLoopsSetTy BoxedLoopsSet;

    /// @brief Loads that need to be invariant during execution.
    polly::InvariantLoadsSetTy RequiredILS;

    /// @brief Map to memory access description for the corresponding LLVM
    ///        instructions.
    MapInsnToMemAcc InsnToMemAcc;

    /// @brief Flag to indicate that this region requires run-time support.
    bool requiresJIT;

    /// @brief List of Parameters values we need to know at run time.
    ParamVec RequiredParams;

    /// @brief Initialize a DetectionContext from scratch.
    DetectionContext(Region &R, BatchAAResults &BAA, bool Verify)
        : CurRegion(R), AST(BAA), Verifying(Verify), Log(&R), hasLoads(false),
          hasStores(false), HasUnknownAccess(false), requiresJIT(false) {}

    /// @brief Initialize a DetectionContext with the data from @p DC.
    DetectionContext(const DetectionContext &&DC)
        : CurRegion(DC.CurRegion), AST(DC.AST.getAliasAnalysis()),
          Verifying(DC.Verifying), Log(DC.Log),
          Accesses(DC.Accesses),
          NonAffineAccesses(DC.NonAffineAccesses),
          ElementSize(DC.ElementSize), hasLoads(DC.hasLoads),
          hasStores(DC.hasStores), HasUnknownAccess(DC.HasUnknownAccess),
          NonAffineSubRegionSet(DC.NonAffineSubRegionSet),
          BoxedLoopsSet(DC.BoxedLoopsSet),
          RequiredILS(DC.RequiredILS),
          requiresJIT(DC.requiresJIT) {
      AST.add(DC.AST);
    }
  };

private:
  //===--------------------------------------------------------------------===//
  JITScopDetection(const JITScopDetection &) = delete;
  const JITScopDetection &operator=(const JITScopDetection &) = delete;

  /// @brief Analysis passes used.
  //@{
  const DominatorTree *DT;
  ScalarEvolution *SE;
  LoopInfo *LI;
  RegionInfo *RI;
  AliasAnalysis *AA;
  BatchAAResults *BAA;
  //@}

  /// @brief Map to remember detection contexts for all regions.
  using DetectionContextMapTy = DenseMap<polly::BBPair, DetectionContext>;
  mutable DetectionContextMapTy DetectionContextMap;

  /// @brief Remove cached results for @p R.
  void removeCachedResults(const Region &R, RegionSet &Cache);

  /// @brief Remove cached results for @p R.
  void removeCachedResults(const Region &R);

  /// @brief Remove cached results for the children of @p R recursively.
  ///
  /// @returns The number of regions erased regions.
  unsigned removeCachedResultsRecursively(const Region &R, RegionSet &Cache);

  /// @brief Add the region @p AR as over approximated sub-region in @p Context.
  ///
  /// @param AR      The non-affine subregion.
  /// @param Context The current detection context.
  ///
  /// @returns True if the subregion can be over approximated, false otherwise.
  bool addOverApproximatedRegion(Region *AR, DetectionContext &Context) const;

  /// @brief Find for a given base pointer terms that hint towards dimension
  ///        sizes of a multi-dimensional array.
  ///
  /// @param Context      The current detection context.
  /// @param BasePointer  A base pointer indicating the virtual array we are
  ///                     interested in.
  SmallVector<const SCEV *, 4>
  getDelinearizationTerms(DetectionContext &Context,
                          const SCEVUnknown *BasePointer) const;

  // Try to expand the region R. If R can be expanded return the expanded
  // region, NULL otherwise.
  Region *expandRegion(Region &R);

  /// Find the Scops in this region tree.
  ///
  /// @param The region tree to scan for scops.
  void findScops(Region &R);

  /// @brief Check if all basic block in the region are valid.
  ///
  /// @param Context The context of scop detection.
  ///
  /// @return True if all blocks in R are valid, false otherwise.
  bool allBlocksValid(DetectionContext &Context) const;

  /// @brief Check if a region has sufficient compute instructions
  ///
  /// This function checks if a region has a non-trivial number of instructions
  /// in each loop. This can be used as an indicator if a loop is worth
  /// optimising.
  ///
  /// @param Context  The context of scop detection.
  /// @param NumLoops The number of loops in the region.
  ///
  /// @return True if region is has sufficient compute instructions,
  ///         false otherwise.
  bool hasSufficientCompute(DetectionContext &Context,
                            int NumAffineLoops) const;

  /// @brief Check if the unique affine loop might be amenable to distribution.
  ///
  /// This function checks if the number of non-trivial blocks in the unique
  /// affine loop in Context.CurRegion is at least two. Thus, the loop might
  /// be amenable to distribution.
  ///
  /// @param Context  The context of SCoP detection.
  ///
  /// @return True only if the affine loop might be distributable.
  bool hasPossiblyDistributableLoop(DetectionContext &Context) const;

  /// @brief Check if a region is profitable to optimize.
  ///
  /// Regions that are unlikely to expose interesting optimization opportunities
  /// are called 'unprofitable' and may be skipped during scop detection.
  ///
  /// @param Context The context of scop detection.
  ///
  /// @return True if region is profitable to optimize, false otherwise.
  bool isProfitableRegion(DetectionContext &Context) const;

  /// @brief Check if a region is a Scop.
  ///
  /// @param Context The context of scop detection.
  ///
  /// @return True if R is a Scop, false otherwise.
  bool isValidRegion(DetectionContext &Context) const;

  /// @brief Check if an intrinsic call can be part of a Scop.
  ///
  /// @param II      The intrinsic call instruction to check.
  /// @param Context The current detection context.
  ///
  /// @return True if the call instruction is valid, false otherwise.
  bool isValidIntrinsicInst(IntrinsicInst &II, DetectionContext &Context) const;

  /// @brief Check if a call instruction can be part of a Scop.
  ///
  /// @param CI      The call instruction to check.
  /// @param Context The current detection context.
  ///
  /// @return True if the call instruction is valid, false otherwise.
  bool isValidCallInst(CallInst &CI, DetectionContext &Context) const;

  /// @brief Check if the given loads could be invariant and can be hoisted.
  ///
  /// If true is returned the loads are added to the required invariant loads
  /// contained in the @p Context.
  ///
  /// @param RequiredILS The loads to check.
  /// @param Context     The current detection context.
  ///
  /// @return True if all loads can be assumed invariant.
  bool onlyValidRequiredInvariantLoads(polly::InvariantLoadsSetTy &RequiredILS,
                                       DetectionContext &Context) const;

  /// @brief Check if a value is invariant in the region Reg.
  ///
  /// @param Val Value to check for invariance.
  /// @param Reg The region to consider for the invariance of Val.
  ///
  /// @return True if the value represented by Val is invariant in the region
  ///         identified by Reg.
  bool isInvariant(const Value &Val, const Region &Reg) const;

  /// @brief Check if the memory access caused by @p Inst is valid.
  ///
  /// @param Inst    The access instruction.
  /// @param AF      The access function.
  /// @param BP      The access base pointer.
  /// @param Context The current detection context.
  bool isValidAccess(Instruction *Inst, const SCEV *AF, const SCEVUnknown *BP,
                     DetectionContext &Context) const;

  /// @brief Check if a memory access can be part of a Scop.
  ///
  /// @param Inst The instruction accessing the memory.
  /// @param Context The context of scop detection.
  ///
  /// @return True if the memory access is valid, false otherwise.
  bool isValidMemoryAccess(polly::MemAccInst Inst, DetectionContext &Context) const;

  /// @brief Check if an instruction has any non trivial scalar dependencies
  ///        as part of a Scop.
  ///
  /// @param Inst The instruction to check.
  /// @param RefRegion The region in respect to which we check the access
  ///                  function.
  ///
  /// @return True if the instruction has scalar dependences, false otherwise.
  bool hasScalarDependency(Instruction &Inst, Region &RefRegion) const;

  /// @brief Check if an instruction can be part of a Scop.
  ///
  /// @param Inst The instruction to check.
  /// @param Context The context of scop detection.
  ///
  /// @return True if the instruction is valid, false otherwise.
  bool isValidInstruction(Instruction &Inst, DetectionContext &Context) const;

  /// @brief Check if the switch @p SI with condition @p Condition is valid.
  ///
  /// @param BB           The block to check.
  /// @param SI           The switch to check.
  /// @param Condition    The switch condition.
  /// @param IsLoopBranch Flag to indicate the branch is a loop exit/latch.
  /// @param Context      The context of scop detection.
  ///
  /// @return True if the branch @p BI is valid.
  bool isValidSwitch(BasicBlock &BB, SwitchInst *SI, Value *Condition,
                     bool IsLoopBranch, DetectionContext &Context) const;

  /// @brief Check if the branch @p BI with condition @p Condition is valid.
  ///
  /// @param BB           The block to check.
  /// @param BI           The branch to check.
  /// @param Condition    The branch condition.
  /// @param IsLoopBranch Flag to indicate the branch is a loop exit/latch.
  /// @param Context      The context of scop detection.
  ///
  /// @return True if the branch @p BI is valid.
  bool isValidBranch(BasicBlock &BB, BranchInst *BI, Value *Condition,
                     bool IsLoopBranch, DetectionContext &Context) const;

  /// @brief Check if the SCEV @p S is affine in the current @p Context.
  ///
  /// This will also use a heuristic to decide if we want to require loads to be
  /// invariant to make the expression affine or if we want to treat is as
  /// non-affine.
  ///
  /// @param S           The expression to be checked.
  /// @param Scope       The loop nest in which @p S is used.
  /// @param Context     The context of scop detection.
  /// @param BaseAddress The base address of the expression @p S (if any).
  bool isAffine(const SCEV *S, Loop *Scope, DetectionContext &Context,
                Value *BaseAddress = nullptr) const;

  /// @brief Check if the control flow in a basic block is valid.
  ///
  /// This function checks if a certain basic block is terminated by a
  /// Terminator instruction we can handle or, if this is not the case,
  /// registers this basic block as the start of a non-affine region.
  ///
  /// This function optionally allows unreachable statements.
  ///
  /// @param BB               The BB to check the control flow.
  /// @param IsLoopBranch     Flag to indicate the branch is a loop exit/latch.
  //  @param AllowUnreachable Allow unreachable statements.
  /// @param Context          The context of scop detection.
  ///
  /// @return True if the BB contains only valid control flow.
  bool isValidCFG(BasicBlock &BB, bool IsLoopBranch, bool AllowUnreachable,
                  DetectionContext &Context) const;

  /// @brief Is a loop valid with respect to a given region.
  ///
  /// @param L The loop to check.
  /// @param Context The context of scop detection.
  ///
  /// @return True if the loop is valid in the region.
  bool isValidLoop(Loop *L, DetectionContext &Context) const;

  /// @brief Count the number of beneficial loops in @p R.
  ///
  /// @param R The region to check
  int countBeneficialLoops(Region *R) const;

  /// @brief Check if the function @p F is marked as invalid.
  ///
  /// @note An OpenMP subfunction will be marked as invalid.
  bool isValidFunction(llvm::Function &F);

  /// @brief Can ISL compute the trip count of a loop.
  ///
  /// @param L The loop to check.
  /// @param Context The context of scop detection.
  ///
  /// @return True if ISL can compute the trip count of the loop.
  bool canUseISLTripCount(Loop *L, DetectionContext &Context) const;

  /// @brief Print the locations of all detected scops.
  void printLocations(llvm::Function &F);

  /// @brief Check if a region is reducible or not.
  ///
  /// @param Region The region to check.
  /// @param DbgLoc Parameter to save the location of instruction that
  ///               causes irregular control flow if the region is irreducible.
  ///
  /// @return True if R is reducible, false otherwise.
  bool isReducibleRegion(Region &R, DebugLoc &DbgLoc) const;

  /// @brief Track diagnostics for invalid scops.
  ///
  /// @param Context The context of scop detection.
  /// @param Assert Throw an assert in verify mode or not.
  /// @param Args Argument list that gets passed to the constructor of RR.
  template <class RR, typename... Args>
  inline bool invalid(DetectionContext &Context, bool Assert,
                      Args &&... Arguments) const;

public:
  static char ID;
  explicit JITScopDetection();

  /// @brief Get the RegionInfo stored in this pass.
  ///
  /// This was added to give the DOT printer easy access to this information.
  RegionInfo *getRI() const { return RI; }

  /// @brief Get the LoopInfo stored in this pass.
  LoopInfo *getLI() const { return LI; }

  /// @brief Is the region is the maximum region of a Scop?
  ///
  /// @param R The Region to test if it is maximum.
  /// @param Verify Rerun the scop detection to verify SCoP was not invalidated
  ///               meanwhile.
  ///
  /// @return Return true if R is the maximum Region in a Scop, false otherwise.
  bool isMaxRegionInScop(const Region &R, bool Verify = true) const;

  /// @brief Return the detection context for @p R, nullptr if @p R was invalid.
  DetectionContext *getDetectionContext(const Region *R) const;

  /// @brief Return the set of required invariant loads for @p R.
  const polly::InvariantLoadsSetTy *getRequiredInvariantLoads(const Region *R) const;

  /// @brief Return the set of rejection causes for @p R.
  const polly::RejectLog *lookupRejectionLog(const Region *R) const;

  /// @brief Return true if @p SubR is a non-affine subregion in @p ScopR.
  bool isNonAffineSubRegion(const Region *SubR, const Region *ScopR) const;

  /// @brief Get a message why a region is invalid
  ///
  /// @param R The region for which we get the error message
  ///
  /// @return The error or "" if no error appeared.
  std::string regionIsInvalidBecause(const Region *R) const;

  /// @name Maximum Region In Scops Iterators
  ///
  /// These iterators iterator over all maximum region in Scops of this
  /// function.
  //@{
  typedef RegionSet::iterator iterator;
  typedef RegionSet::const_iterator const_iterator;

  iterator begin() { return ValidRegions.begin(); }
  iterator end() { return ValidRegions.end(); }

  const_iterator begin() const { return ValidRegions.begin(); }
  const_iterator end() const { return ValidRegions.end(); }
  //@}

  /// @brief Mark the function as invalid so we will not extract any scop from
  ///        the function.
  ///
  /// @param F The function to mark as invalid.
  void markFunctionAsInvalid(Function *F) const;

  /// @brief Verify if all valid Regions in this Function are still valid
  /// after some transformations.
  void verifyAnalysis() const override;

  /// @brief Verify if R is still a valid part of Scop after some
  /// transformations.
  ///
  /// @param R The Region to verify.
  void verifyRegion(const Region &R) const;

  /// @name FunctionPass interface
  //@{
  virtual void getAnalysisUsage(AnalysisUsage &AU) const override;
  virtual void releaseMemory() override;
  virtual bool runOnFunction(Function &F) override;
  virtual void print(raw_ostream &OS, const Module *) const override;
  //@}
};

Pass *createScopDetectionPass();
} // end namespace proteus

namespace llvm {
class PassRegistry;
void initializeJITScopDetectionPass(llvm::PassRegistry &);
} // namespace llvm
#endif // proteus_SCOPDETECTION_H
