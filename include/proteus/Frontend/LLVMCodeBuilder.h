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
  CodeBuilderKind getBackendKind() const override {
    return CodeBuilderKind::LLVM;
  }

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
  IRValue *createArith(ArithOp Op, IRValue *LHS, IRValue *RHS,
                       IRType Ty) override;

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
  IRValue *createCmp(CmpOp Op, IRValue *LHS, IRValue *RHS, IRType Ty) override;

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
  IRValue *createCast(IRValue *V, IRType FromTy, IRType ToTy) override;
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
  VarAlloc getElementPtr(IRValue *Base, IRType BaseTy, IRValue *Index,
                         IRType ElemTy) override;
  VarAlloc getElementPtr(IRValue *Base, IRType BaseTy, size_t Index,
                         IRType ElemTy) override;

  // -----------------------------------------------------------------------
  // CodeBuilder overrides — calls.
  // -----------------------------------------------------------------------
  IRValue *createCall(const std::string &FName, IRType RetTy,
                      const std::vector<IRType> &ArgTys,
                      const std::vector<IRValue *> &Args) override;
  IRValue *createCall(const std::string &FName, IRType RetTy) override;
  IRValue *emitIntrinsic(const std::string &Name, IRType RetTy,
                         const std::vector<IRValue *> &Args) override;
  IRValue *emitBuiltin(const std::string &Name, IRType RetTy,
                       const std::vector<IRValue *> &Args) override;

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

  // -----------------------------------------------------------------------
  // Insertion point management (LLVM-specific extensions).
  // -----------------------------------------------------------------------
  void setInsertPoint(llvm::BasicBlock *BB);
  void setInsertPointBegin(llvm::BasicBlock *BB);
  llvm::BasicBlock *getInsertBlock();

  // -----------------------------------------------------------------------
  // Private low-level arithmetic dispatch (called by createArith).
  // -----------------------------------------------------------------------
  IRValue *createAdd(IRValue *LHS, IRValue *RHS);
  IRValue *createFAdd(IRValue *LHS, IRValue *RHS);
  IRValue *createSub(IRValue *LHS, IRValue *RHS);
  IRValue *createFSub(IRValue *LHS, IRValue *RHS);
  IRValue *createMul(IRValue *LHS, IRValue *RHS);
  IRValue *createFMul(IRValue *LHS, IRValue *RHS);
  IRValue *createUDiv(IRValue *LHS, IRValue *RHS);
  IRValue *createSDiv(IRValue *LHS, IRValue *RHS);
  IRValue *createFDiv(IRValue *LHS, IRValue *RHS);
  IRValue *createURem(IRValue *LHS, IRValue *RHS);
  IRValue *createSRem(IRValue *LHS, IRValue *RHS);
  IRValue *createFRem(IRValue *LHS, IRValue *RHS);

  // -----------------------------------------------------------------------
  // Private low-level comparison dispatch (called by createCmp).
  // -----------------------------------------------------------------------
  IRValue *createICmpEQ(IRValue *LHS, IRValue *RHS);
  IRValue *createICmpNE(IRValue *LHS, IRValue *RHS);
  IRValue *createICmpSLT(IRValue *LHS, IRValue *RHS);
  IRValue *createICmpSGT(IRValue *LHS, IRValue *RHS);
  IRValue *createICmpSGE(IRValue *LHS, IRValue *RHS);
  IRValue *createICmpSLE(IRValue *LHS, IRValue *RHS);
  IRValue *createICmpUGT(IRValue *LHS, IRValue *RHS);
  IRValue *createICmpUGE(IRValue *LHS, IRValue *RHS);
  IRValue *createICmpULT(IRValue *LHS, IRValue *RHS);
  IRValue *createICmpULE(IRValue *LHS, IRValue *RHS);
  IRValue *createFCmpOEQ(IRValue *LHS, IRValue *RHS);
  IRValue *createFCmpONE(IRValue *LHS, IRValue *RHS);
  IRValue *createFCmpOLT(IRValue *LHS, IRValue *RHS);
  IRValue *createFCmpOLE(IRValue *LHS, IRValue *RHS);
  IRValue *createFCmpOGT(IRValue *LHS, IRValue *RHS);
  IRValue *createFCmpOGE(IRValue *LHS, IRValue *RHS);
  IRValue *createFCmpULT(IRValue *LHS, IRValue *RHS);
  IRValue *createFCmpULE(IRValue *LHS, IRValue *RHS);

  // -----------------------------------------------------------------------
  // Private low-level cast dispatch (called by createCast).
  // -----------------------------------------------------------------------
  IRValue *createIntCast(IRValue *V, IRType DestTy, bool IsSigned);
  IRValue *createFPCast(IRValue *V, IRType DestTy);
  IRValue *createSIToFP(IRValue *V, IRType DestTy);
  IRValue *createUIToFP(IRValue *V, IRType DestTy);
  IRValue *createFPToSI(IRValue *V, IRType DestTy);
  IRValue *createFPToUI(IRValue *V, IRType DestTy);

  // -----------------------------------------------------------------------
  // Private GEP helpers (called by getElementPtr).
  // -----------------------------------------------------------------------
  IRValue *createInBoundsGEP(IRType Ty, IRValue *Ptr,
                             const std::vector<IRValue *> &IdxList,
                             const std::string &Name = "");
  // NOLINTNEXTLINE
  IRValue *createConstInBoundsGEP1_64(IRType Ty, IRValue *Ptr, size_t Idx);
  // NOLINTNEXTLINE
  IRValue *createConstInBoundsGEP2_64(IRType Ty, IRValue *Ptr, size_t Idx0,
                                      size_t Idx1);
  unsigned getAddressSpaceFromValue(IRValue *PtrVal);
};

} // namespace proteus

#endif // PROTEUS_FRONTEND_LLVM_CODE_BUILDER_H
