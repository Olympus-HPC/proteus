#ifndef PROTEUS_FRONTEND_LLVM_CODE_BUILDER_H
#define PROTEUS_FRONTEND_LLVM_CODE_BUILDER_H

#include "proteus/AddressSpace.h"
#include "proteus/Error.h"
#include "proteus/Frontend/TargetModel.h"
#include "proteus/Frontend/VarStorage.h"

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
class Value;
} // namespace llvm

namespace proteus {

enum class ScopeKind { FUNCTION, IF, FOR, WHILE };

inline std::string toString(ScopeKind Kind) {
  switch (Kind) {
  case ScopeKind::FUNCTION:
    return "FUNCTION";
  case ScopeKind::IF:
    return "IF";
  case ScopeKind::FOR:
    return "FOR";
  case ScopeKind::WHILE:
    return "WHILE";
  default:
    reportFatalError("Unsupported Kind " +
                     std::to_string(static_cast<int>(Kind)));
  }
}

/// LLVMCodeBuilder encapsulates LLVM IR generation using IRBuilder.
/// It manages insertion points, scopes, and provides methods for
/// creating LLVM IR instructions.
class LLVMCodeBuilder {
public:
  /// Construct as owner of LLVMContext and Module.
  LLVMCodeBuilder(std::unique_ptr<llvm::LLVMContext> Ctx,
                  std::unique_ptr<llvm::Module> Mod,
                  TargetModelType TM = TargetModelType::HOST);
  ~LLVMCodeBuilder();

  TargetModelType getTargetModel() const { return TargetModel; }

  LLVMCodeBuilder(const LLVMCodeBuilder &) = delete;
  LLVMCodeBuilder &operator=(const LLVMCodeBuilder &) = delete;

  /// Get the underlying IRBuilderBase (e.g., for VarStorage construction).
  llvm::IRBuilderBase &getIRBuilder();

  llvm::Function &getFunction();
  llvm::Module &getModule();
  llvm::LLVMContext &getContext();

  /// Create a new Function in the owned Module, initialise its entry block,
  /// and make it the current active function.  Only valid on the owning
  /// constructor variant.
  llvm::Function *addFunction(const std::string &Name, llvm::Type *RetTy,
                              const std::vector<llvm::Type *> &ArgTys);

  /// Transfer ownership of the LLVMContext (leaves internal pointer null).
  std::unique_ptr<llvm::LLVMContext> takeLLVMContext();
  /// Transfer ownership of the Module (leaves internal pointer null).
  std::unique_ptr<llvm::Module> takeModule();

  // Insertion point management.
  void setInsertPoint(llvm::BasicBlock *BB);
  void setInsertPointBegin(llvm::BasicBlock *BB);
  void setInsertPointAtEntry();
  void clearInsertPoint();
  llvm::BasicBlock *getInsertBlock();

  // Basic block operations.
  std::tuple<llvm::BasicBlock *, llvm::BasicBlock *> splitCurrentBlock();
  llvm::BasicBlock *createBasicBlock(const std::string &Name = "",
                                     llvm::BasicBlock *InsertBefore = nullptr);
  void eraseTerminator(llvm::BasicBlock *BB);
  llvm::BasicBlock *getUniqueSuccessor(llvm::BasicBlock *BB);

  // Scope management.
  void pushScope(const char *File, int Line, ScopeKind Kind,
                 llvm::BasicBlock *NextBlock);

  // High-level scope operations.
  /// Begin code generation for Fn.  Sets the active function to Fn.
  void beginFunction(llvm::Function &Fn, const char *File, int Line);
  void endFunction();
  void beginIf(llvm::Value *Cond, const char *File, int Line);
  void endIf();
  /// IterSlot : alloca holding the loop iterator.
  /// IterTy   : value type of the iterator (must be an integer type).
  /// InitVal  : initial value to store into IterSlot.
  /// UpperBoundVal : exclusive upper bound for the loop condition.
  /// IncVal   : increment added to the iterator on each iteration.
  /// IsSigned : true → ICmpSLT, false → ICmpULT.
  void beginFor(llvm::Value *IterSlot, llvm::Type *IterTy, llvm::Value *InitVal,
                llvm::Value *UpperBoundVal, llvm::Value *IncVal, bool IsSigned,
                const char *File, int Line);
  void endFor();
  /// CondFn : callable that emits the condition IR at the current insert point
  ///          and returns the resulting i1 Value (true → continue loop).
  void beginWhile(std::function<llvm::Value *()> CondFn, const char *File,
                  int Line);
  void endWhile();

  // Arithmetic operations.
  llvm::Value *createAdd(llvm::Value *LHS, llvm::Value *RHS);
  llvm::Value *createFAdd(llvm::Value *LHS, llvm::Value *RHS);
  llvm::Value *createSub(llvm::Value *LHS, llvm::Value *RHS);
  llvm::Value *createFSub(llvm::Value *LHS, llvm::Value *RHS);
  llvm::Value *createMul(llvm::Value *LHS, llvm::Value *RHS);
  llvm::Value *createFMul(llvm::Value *LHS, llvm::Value *RHS);
  llvm::Value *createUDiv(llvm::Value *LHS, llvm::Value *RHS);
  llvm::Value *createSDiv(llvm::Value *LHS, llvm::Value *RHS);
  llvm::Value *createFDiv(llvm::Value *LHS, llvm::Value *RHS);
  llvm::Value *createURem(llvm::Value *LHS, llvm::Value *RHS);
  llvm::Value *createSRem(llvm::Value *LHS, llvm::Value *RHS);
  llvm::Value *createFRem(llvm::Value *LHS, llvm::Value *RHS);

  // Atomic operations.
  llvm::Value *createAtomicAdd(llvm::Value *Addr, llvm::Value *Val);
  llvm::Value *createAtomicSub(llvm::Value *Addr, llvm::Value *Val);
  llvm::Value *createAtomicMax(llvm::Value *Addr, llvm::Value *Val);
  llvm::Value *createAtomicMin(llvm::Value *Addr, llvm::Value *Val);

  // Comparison operations.
  llvm::Value *createICmpEQ(llvm::Value *LHS, llvm::Value *RHS);
  llvm::Value *createICmpNE(llvm::Value *LHS, llvm::Value *RHS);
  llvm::Value *createICmpSLT(llvm::Value *LHS, llvm::Value *RHS);
  llvm::Value *createICmpSGT(llvm::Value *LHS, llvm::Value *RHS);
  llvm::Value *createICmpSGE(llvm::Value *LHS, llvm::Value *RHS);
  llvm::Value *createICmpSLE(llvm::Value *LHS, llvm::Value *RHS);
  llvm::Value *createICmpUGT(llvm::Value *LHS, llvm::Value *RHS);
  llvm::Value *createICmpUGE(llvm::Value *LHS, llvm::Value *RHS);
  llvm::Value *createICmpULT(llvm::Value *LHS, llvm::Value *RHS);
  llvm::Value *createICmpULE(llvm::Value *LHS, llvm::Value *RHS);
  llvm::Value *createFCmpOEQ(llvm::Value *LHS, llvm::Value *RHS);
  llvm::Value *createFCmpONE(llvm::Value *LHS, llvm::Value *RHS);
  llvm::Value *createFCmpOLT(llvm::Value *LHS, llvm::Value *RHS);
  llvm::Value *createFCmpOLE(llvm::Value *LHS, llvm::Value *RHS);
  llvm::Value *createFCmpOGT(llvm::Value *LHS, llvm::Value *RHS);
  llvm::Value *createFCmpOGE(llvm::Value *LHS, llvm::Value *RHS);
  llvm::Value *createFCmpULT(llvm::Value *LHS, llvm::Value *RHS);
  llvm::Value *createFCmpULE(llvm::Value *LHS, llvm::Value *RHS);

  // Logical operations.
  llvm::Value *createAnd(llvm::Value *LHS, llvm::Value *RHS);
  llvm::Value *createOr(llvm::Value *LHS, llvm::Value *RHS);
  llvm::Value *createXor(llvm::Value *LHS, llvm::Value *RHS);
  llvm::Value *createNot(llvm::Value *Val);

  // Load/Store operations.
  llvm::Value *createLoad(llvm::Type *Ty, llvm::Value *Ptr,
                          const std::string &Name = "");
  void createStore(llvm::Value *Val, llvm::Value *Ptr);

  // Control flow operations.
  void createBr(llvm::BasicBlock *Dest);
  void createCondBr(llvm::Value *Cond, llvm::BasicBlock *True,
                    llvm::BasicBlock *False);
  void createRetVoid();
  void createRet(llvm::Value *V);

  // Cast operations.
  llvm::Value *createIntCast(llvm::Value *V, llvm::Type *DestTy, bool IsSigned);
  llvm::Value *createFPCast(llvm::Value *V, llvm::Type *DestTy);
  llvm::Value *createSIToFP(llvm::Value *V, llvm::Type *DestTy);
  llvm::Value *createUIToFP(llvm::Value *V, llvm::Type *DestTy);
  llvm::Value *createFPToSI(llvm::Value *V, llvm::Type *DestTy);
  llvm::Value *createFPToUI(llvm::Value *V, llvm::Type *DestTy);
  llvm::Value *createBitCast(llvm::Value *V, llvm::Type *DestTy);
  llvm::Value *createZExt(llvm::Value *V, llvm::Type *DestTy);

  // Constant creation.
  llvm::Value *getConstantInt(llvm::Type *Ty, uint64_t Val);
  llvm::Value *getConstantFP(llvm::Type *Ty, double Val);

  // GEP operations.
  llvm::Value *createInBoundsGEP(llvm::Type *Ty, llvm::Value *Ptr,
                                 const std::vector<llvm::Value *> IdxList,
                                 const std::string &Name = "");
  // NOLINTNEXTLINE
  llvm::Value *createConstInBoundsGEP1_64(llvm::Type *Ty, llvm::Value *Ptr,
                                          size_t Idx);
  // NOLINTNEXTLINE
  llvm::Value *createConstInBoundsGEP2_64(llvm::Type *Ty, llvm::Value *Ptr,
                                          size_t Idx0, size_t Idx1);

  // Type accessors.
  llvm::Type *getPointerType(llvm::Type *ElemTy, unsigned AS);
  llvm::Type *getPointerTypeUnqual(llvm::Type *ElemTy);
  llvm::Type *getInt16Ty();
  llvm::Type *getInt32Ty();
  llvm::Type *getInt64Ty();
  llvm::Type *getFloatTy();

  // Type queries.
  unsigned getAddressSpace(llvm::Type *Ty);
  unsigned getAddressSpaceFromValue(llvm::Value *PtrVal);
  bool isIntegerTy(llvm::Type *Ty);
  bool isFloatingPointTy(llvm::Type *Ty);

  // Call operations.
  llvm::Value *createCall(const std::string &FName, llvm::Type *RetTy,
                          const std::vector<llvm::Type *> &ArgTys,
                          const std::vector<llvm::Value *> &Args);
  llvm::Value *createCall(const std::string &FName, llvm::Type *RetTy);

  // Alloca/array emission.
  llvm::Value *emitAlloca(llvm::Type *Ty, const std::string &Name,
                          AddressSpace AS = AddressSpace::DEFAULT);
  llvm::Value *emitArrayCreate(llvm::Type *Ty, AddressSpace AT,
                               const std::string &Name);
  std::unique_ptr<ScalarStorage> createScalarStorage(const std::string &Name,
                                                     llvm::Type *AllocaTy);
  std::unique_ptr<PointerStorage> createPointerStorage(const std::string &Name,
                                                       llvm::Type *AllocaTy,
                                                       llvm::Type *ElemTy);
  std::unique_ptr<ArrayStorage> createArrayStorage(const std::string &Name,
                                                   AddressSpace AS,
                                                   llvm::Type *ArrTy);
#if defined(PROTEUS_ENABLE_CUDA) || defined(PROTEUS_ENABLE_HIP)
  void setKernel(llvm::Function &Fn);
  void setLaunchBoundsForKernel(llvm::Function &Fn, int MaxThreadsPerBlock,
                                int MinBlocksPerSM);
#endif

private:
  struct Impl;
  std::unique_ptr<Impl> PImpl;
  llvm::Function *F;
  TargetModelType TargetModel;
};

} // namespace proteus

#endif // PROTEUS_FRONTEND_LLVM_CODE_BUILDER_H
