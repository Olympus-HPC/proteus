#ifndef PROTEUS_FRONTEND_LLVM_CODE_BUILDER_H
#define PROTEUS_FRONTEND_LLVM_CODE_BUILDER_H

#include "proteus/Frontend/CodeBuilder.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace llvm {
class AllocaInst;
class ArrayType;
class BasicBlock;
class Function;
class IRBuilderBase;
class LLVMContext;
class Module;
class Type;
} // namespace llvm

namespace proteus {

/// LLVMCodeBuilder encapsulates LLVM IR generation using IRBuilder.
/// It manages insertion points, scopes, and provides methods for
/// creating LLVM IR instructions.
class LLVMCodeBuilder : public CodeBuilder {
public:
  /// Construct as owner of LLVMContext and Module.
  LLVMCodeBuilder(std::unique_ptr<llvm::LLVMContext> Ctx,
                  std::unique_ptr<llvm::Module> Mod,
                  TargetModelType TM = TargetModelType::HOST);
  ~LLVMCodeBuilder() override;

  TargetModelType getTargetModel() const override { return TargetModel; }

  LLVMCodeBuilder(const LLVMCodeBuilder &) = delete;
  LLVMCodeBuilder &operator=(const LLVMCodeBuilder &) = delete;

  // -----------------------------------------------------------------------
  // LLVM-specific (non-virtual) accessors.
  // -----------------------------------------------------------------------

  /// Get the underlying IRBuilderBase (internal use only).
  llvm::IRBuilderBase &getIRBuilder();

  llvm::Function &getFunction();
  llvm::Module &getModule();
  llvm::LLVMContext &getContext();

  /// Transfer ownership of the LLVMContext (leaves internal pointer null).
  std::unique_ptr<llvm::LLVMContext> takeLLVMContext();
  /// Transfer ownership of the Module (leaves internal pointer null).
  std::unique_ptr<llvm::Module> takeModule();

  // -----------------------------------------------------------------------
  // Insertion point management (LLVM-specific extensions).
  // -----------------------------------------------------------------------
  void setInsertPoint(llvm::BasicBlock *BB);
  void setInsertPointBegin(llvm::BasicBlock *BB);
  llvm::BasicBlock *getInsertBlock();

  // -----------------------------------------------------------------------
  // Basic block operations.
  // -----------------------------------------------------------------------
  std::tuple<llvm::BasicBlock *, llvm::BasicBlock *> splitCurrentBlock();
  llvm::BasicBlock *createBasicBlock(const std::string &Name = "",
                                     llvm::BasicBlock *InsertBefore = nullptr);
  void eraseTerminator(llvm::BasicBlock *BB);
  llvm::BasicBlock *getUniqueSuccessor(llvm::BasicBlock *BB);

  // -----------------------------------------------------------------------
  // Scope management.
  // -----------------------------------------------------------------------
  void pushScope(const char *File, int Line, ScopeKind Kind,
                 llvm::BasicBlock *NextBlock);

  // -----------------------------------------------------------------------
  // Control flow (LLVM-specific overloads / extensions).
  // -----------------------------------------------------------------------
  void createBr(llvm::BasicBlock *Dest);
  void createCondBr(IRValue *Cond, llvm::BasicBlock *True,
                    llvm::BasicBlock *False);

  // -----------------------------------------------------------------------
  // Type accessors.
  // -----------------------------------------------------------------------
  llvm::Type *getPointerType(llvm::Type *ElemTy, unsigned AS);
  llvm::Type *getPointerTypeUnqual(llvm::Type *ElemTy);
  llvm::Type *getInt16Ty();
  llvm::Type *getInt32Ty();
  llvm::Type *getInt64Ty();
  llvm::Type *getFloatTy();

  // -----------------------------------------------------------------------
  // Type queries.
  // -----------------------------------------------------------------------
  unsigned getAddressSpace(llvm::Type *Ty);
  bool isIntegerTy(llvm::Type *Ty);
  bool isFloatingPointTy(llvm::Type *Ty);

  // -----------------------------------------------------------------------
  // Alloca/array emission.
  // -----------------------------------------------------------------------
  IRValue *emitAlloca(llvm::Type *Ty, const std::string &Name,
                      AddressSpace AS = AddressSpace::DEFAULT);
  IRValue *emitArrayCreate(llvm::Type *Ty, AddressSpace AT,
                           const std::string &Name);

  // -----------------------------------------------------------------------
  // CodeBuilder overrides — function management.
  // -----------------------------------------------------------------------
  IRFunction *addFunction(const std::string &Name, IRType RetTy,
                          const std::vector<IRType> &ArgTys) override;
  void setFunctionName(IRFunction *F, const std::string &Name) override;
  IRValue *getArg(IRFunction *F, size_t Idx) override;
  void beginFunction(IRFunction *F, const char *File, int Line) override;
  void endFunction() override;

  // -----------------------------------------------------------------------
  // CodeBuilder overrides — insertion point management.
  // -----------------------------------------------------------------------
  void setInsertPointAtEntry() override;
  void clearInsertPoint() override;

  // -----------------------------------------------------------------------
  // CodeBuilder overrides — high-level control flow.
  // -----------------------------------------------------------------------
  void beginIf(IRValue *Cond, const char *File, int Line) override;
  void endIf() override;
  void beginFor(IRValue *IterSlot, IRType IterTy, IRValue *InitVal,
                IRValue *UpperBoundVal, IRValue *IncVal, bool IsSigned,
                const char *File, int Line, LoopHints Hints = {}) override;
  void endFor() override;
  void beginWhile(std::function<IRValue *()> CondFn, const char *File,
                  int Line) override;
  void endWhile() override;
  void createRetVoid() override;
  void createRet(IRValue *V) override;

  // -----------------------------------------------------------------------
  // CodeBuilder overrides — arithmetic.
  // -----------------------------------------------------------------------
  IRValue *createAdd(IRValue *LHS, IRValue *RHS) override;
  IRValue *createFAdd(IRValue *LHS, IRValue *RHS) override;
  IRValue *createSub(IRValue *LHS, IRValue *RHS) override;
  IRValue *createFSub(IRValue *LHS, IRValue *RHS) override;
  IRValue *createMul(IRValue *LHS, IRValue *RHS) override;
  IRValue *createFMul(IRValue *LHS, IRValue *RHS) override;
  IRValue *createUDiv(IRValue *LHS, IRValue *RHS) override;
  IRValue *createSDiv(IRValue *LHS, IRValue *RHS) override;
  IRValue *createFDiv(IRValue *LHS, IRValue *RHS) override;
  IRValue *createURem(IRValue *LHS, IRValue *RHS) override;
  IRValue *createSRem(IRValue *LHS, IRValue *RHS) override;
  IRValue *createFRem(IRValue *LHS, IRValue *RHS) override;

  // -----------------------------------------------------------------------
  // CodeBuilder overrides — atomics.
  // -----------------------------------------------------------------------
  IRValue *createAtomicAdd(IRValue *Addr, IRValue *Val) override;
  IRValue *createAtomicSub(IRValue *Addr, IRValue *Val) override;
  IRValue *createAtomicMax(IRValue *Addr, IRValue *Val) override;
  IRValue *createAtomicMin(IRValue *Addr, IRValue *Val) override;

  // -----------------------------------------------------------------------
  // CodeBuilder overrides — comparisons.
  // -----------------------------------------------------------------------
  IRValue *createICmpEQ(IRValue *LHS, IRValue *RHS) override;
  IRValue *createICmpNE(IRValue *LHS, IRValue *RHS) override;
  IRValue *createICmpSLT(IRValue *LHS, IRValue *RHS) override;
  IRValue *createICmpSGT(IRValue *LHS, IRValue *RHS) override;
  IRValue *createICmpSGE(IRValue *LHS, IRValue *RHS) override;
  IRValue *createICmpSLE(IRValue *LHS, IRValue *RHS) override;
  IRValue *createICmpUGT(IRValue *LHS, IRValue *RHS) override;
  IRValue *createICmpUGE(IRValue *LHS, IRValue *RHS) override;
  IRValue *createICmpULT(IRValue *LHS, IRValue *RHS) override;
  IRValue *createICmpULE(IRValue *LHS, IRValue *RHS) override;
  IRValue *createFCmpOEQ(IRValue *LHS, IRValue *RHS) override;
  IRValue *createFCmpONE(IRValue *LHS, IRValue *RHS) override;
  IRValue *createFCmpOLT(IRValue *LHS, IRValue *RHS) override;
  IRValue *createFCmpOLE(IRValue *LHS, IRValue *RHS) override;
  IRValue *createFCmpOGT(IRValue *LHS, IRValue *RHS) override;
  IRValue *createFCmpOGE(IRValue *LHS, IRValue *RHS) override;
  IRValue *createFCmpULT(IRValue *LHS, IRValue *RHS) override;
  IRValue *createFCmpULE(IRValue *LHS, IRValue *RHS) override;

  // -----------------------------------------------------------------------
  // CodeBuilder overrides — logical.
  // -----------------------------------------------------------------------
  IRValue *createAnd(IRValue *LHS, IRValue *RHS) override;
  IRValue *createOr(IRValue *LHS, IRValue *RHS) override;
  IRValue *createXor(IRValue *LHS, IRValue *RHS) override;
  IRValue *createNot(IRValue *Val) override;

  // -----------------------------------------------------------------------
  // CodeBuilder overrides — load/store.
  // -----------------------------------------------------------------------
  IRValue *createLoad(IRType Ty, IRValue *Ptr,
                      const std::string &Name = "") override;
  void createStore(IRValue *Val, IRValue *Ptr) override;

  // -----------------------------------------------------------------------
  // CodeBuilder overrides — casts.
  // -----------------------------------------------------------------------
  IRValue *createIntCast(IRValue *V, IRType DestTy, bool IsSigned) override;
  IRValue *createFPCast(IRValue *V, IRType DestTy) override;
  IRValue *createSIToFP(IRValue *V, IRType DestTy) override;
  IRValue *createUIToFP(IRValue *V, IRType DestTy) override;
  IRValue *createFPToSI(IRValue *V, IRType DestTy) override;
  IRValue *createFPToUI(IRValue *V, IRType DestTy) override;
  IRValue *createBitCast(IRValue *V, IRType DestTy) override;
  IRValue *createZExt(IRValue *V, IRType DestTy) override;

  // -----------------------------------------------------------------------
  // CodeBuilder overrides — constants.
  // -----------------------------------------------------------------------
  IRValue *getConstantInt(IRType Ty, uint64_t Val) override;
  IRValue *getConstantFP(IRType Ty, double Val) override;

  // -----------------------------------------------------------------------
  // CodeBuilder overrides — GEP.
  // -----------------------------------------------------------------------
  IRValue *createInBoundsGEP(IRType Ty, IRValue *Ptr,
                             const std::vector<IRValue *> IdxList,
                             const std::string &Name = "") override;
  // NOLINTNEXTLINE
  IRValue *createConstInBoundsGEP1_64(IRType Ty, IRValue *Ptr,
                                      size_t Idx) override;
  // NOLINTNEXTLINE
  IRValue *createConstInBoundsGEP2_64(IRType Ty, IRValue *Ptr, size_t Idx0,
                                      size_t Idx1) override;

  // -----------------------------------------------------------------------
  // CodeBuilder overrides — address-space query.
  // -----------------------------------------------------------------------
  unsigned getAddressSpaceFromValue(IRValue *PtrVal) override;

  // -----------------------------------------------------------------------
  // CodeBuilder overrides — calls.
  // -----------------------------------------------------------------------
  IRValue *createCall(const std::string &FName, IRType RetTy,
                      const std::vector<IRType> &ArgTys,
                      const std::vector<IRValue *> &Args) override;
  IRValue *createCall(const std::string &FName, IRType RetTy) override;

  // -----------------------------------------------------------------------
  // CodeBuilder overrides — storage-aware load/store.
  // -----------------------------------------------------------------------
  IRValue *loadScalar(IRValue *Slot, IRType ValueTy) override;
  void storeScalar(IRValue *Slot, IRValue *Val) override;
  IRValue *loadAddress(IRValue *Slot, IRType AllocTy) override;
  void storeAddress(IRValue *Slot, IRValue *Addr) override;
  IRValue *loadFromPointee(IRValue *Slot, IRType AllocTy,
                           IRType ValueTy) override;
  void storeToPointee(IRValue *Slot, IRType AllocTy, IRValue *Val) override;

  // -----------------------------------------------------------------------
  // CodeBuilder overrides — alloc factories.
  // -----------------------------------------------------------------------
  VarAlloc allocScalar(const std::string &Name, IRType ValueTy) override;
  VarAlloc allocPointer(const std::string &Name, IRType ElemTy,
                        unsigned AddrSpace = 0) override;
  VarAlloc allocArray(const std::string &Name, AddressSpace AS, IRType ElemTy,
                      size_t NElem) override;

  // -----------------------------------------------------------------------
  // CodeBuilder overrides — GPU kernel support.
  // -----------------------------------------------------------------------
#if defined(PROTEUS_ENABLE_CUDA) || defined(PROTEUS_ENABLE_HIP)
  void setKernel(IRFunction *F) override;
  void setLaunchBoundsForKernel(IRFunction *F, int MaxThreadsPerBlock,
                                int MinBlocksPerSM) override;
#endif

private:
  struct Impl;
  std::unique_ptr<Impl> PImpl;
  llvm::Function *F;
  TargetModelType TargetModel;

  /// Unwrap an opaque IRFunction back to a raw llvm::Function pointer.
  llvm::Function *unwrapFunction(IRFunction *IRF);
};

} // namespace proteus

#endif // PROTEUS_FRONTEND_LLVM_CODE_BUILDER_H
