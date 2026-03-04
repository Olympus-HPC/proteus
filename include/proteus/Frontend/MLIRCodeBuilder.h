#ifndef PROTEUS_FRONTEND_MLIR_CODE_BUILDER_H
#define PROTEUS_FRONTEND_MLIR_CODE_BUILDER_H

#include "proteus/Frontend/CodeBuilder.h"

#include <memory>
#include <string>
#include <vector>

namespace proteus {

/// MLIRCodeBuilder implements the CodeBuilder interface using MLIR dialects.
/// Host mode emits func/arith/memref, while CUDA/HIP mode emits gpu.module
/// with gpu.func kernels.
///
/// Phase 1b: supports addFunction, getArg, beginFunction/endFunction,
/// allocScalar, loadScalar/storeScalar, createArith, createCast,
/// createRet/createRetVoid, getConstantInt/getConstantFP, and print.
/// All other CodeBuilder methods report a fatal error.
class MLIRCodeBuilder : public CodeBuilder {
public:
  explicit MLIRCodeBuilder(TargetModelType TM = TargetModelType::HOST);
  ~MLIRCodeBuilder() override;

  MLIRCodeBuilder(const MLIRCodeBuilder &) = delete;
  MLIRCodeBuilder &operator=(const MLIRCodeBuilder &) = delete;

  TargetModelType getTargetModel() const override { return TargetModel; }
  CodeBuilderKind getBackendKind() const override {
    return CodeBuilderKind::MLIR;
  }

  /// Dump the MLIR module to llvm::outs().
  void print();

  // -----------------------------------------------------------------------
  // Implemented CodeBuilder overrides (Phase 1b).
  // -----------------------------------------------------------------------

  IRFunction *addFunction(const std::string &Name, IRType RetTy,
                          const std::vector<IRType> &ArgTys,
                          bool IsKernel = false) override;
  void setFunctionName(IRFunction *F, const std::string &Name) override;
  IRValue *getArg(IRFunction *F, size_t Idx) override;
  void beginFunction(IRFunction *F, const char *File, int Line) override;
  void endFunction() override;
  void setInsertPointAtEntry() override;
  void clearInsertPoint() override;
  void createRetVoid() override;
  void createRet(IRValue *V) override;
  IRValue *createArith(ArithOp Op, IRValue *LHS, IRValue *RHS,
                       IRType Ty) override;
  IRValue *createCast(IRValue *V, IRType FromTy, IRType ToTy) override;
  IRValue *getConstantInt(IRType Ty, uint64_t Val) override;
  IRValue *getConstantFP(IRType Ty, double Val) override;
  IRValue *loadScalar(IRValue *Slot, IRType ValueTy) override;
  void storeScalar(IRValue *Slot, IRValue *Val) override;
  VarAlloc allocScalar(const std::string &Name, IRType ValueTy) override;

  // -----------------------------------------------------------------------
  // Stub CodeBuilder overrides (fatal error, to be implemented later).
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
  IRValue *createAtomicAdd(IRValue *Addr, IRValue *Val) override;
  IRValue *createAtomicSub(IRValue *Addr, IRValue *Val) override;
  IRValue *createAtomicMax(IRValue *Addr, IRValue *Val) override;
  IRValue *createAtomicMin(IRValue *Addr, IRValue *Val) override;
  IRValue *createCmp(CmpOp Op, IRValue *LHS, IRValue *RHS, IRType Ty) override;
  IRValue *createAnd(IRValue *LHS, IRValue *RHS) override;
  IRValue *createOr(IRValue *LHS, IRValue *RHS) override;
  IRValue *createXor(IRValue *LHS, IRValue *RHS) override;
  IRValue *createNot(IRValue *Val) override;
  IRValue *createLoad(IRType Ty, IRValue *Ptr,
                      const std::string &Name = "") override;
  void createStore(IRValue *Val, IRValue *Ptr) override;
  IRValue *createBitCast(IRValue *V, IRType DestTy) override;
  IRValue *createZExt(IRValue *V, IRType DestTy) override;
  VarAlloc getElementPtr(IRValue *Base, IRType BaseTy, IRValue *Index,
                         IRType ElemTy) override;
  // NOLINTNEXTLINE
  VarAlloc getElementPtr(IRValue *Base, IRType BaseTy, size_t Index,
                         IRType ElemTy) override;
  IRValue *createCall(const std::string &FName, IRType RetTy,
                      const std::vector<IRType> &ArgTys,
                      const std::vector<IRValue *> &Args) override;
  IRValue *createCall(const std::string &FName, IRType RetTy) override;
  IRValue *emitIntrinsic(const std::string &Name, IRType RetTy,
                         const std::vector<IRValue *> &Args) override;
  IRValue *emitBuiltin(const std::string &Name, IRType RetTy,
                       const std::vector<IRValue *> &Args) override;
  IRValue *loadAddress(IRValue *Slot, IRType AllocTy) override;
  void storeAddress(IRValue *Slot, IRValue *Addr) override;
  IRValue *loadFromPointee(IRValue *Slot, IRType AllocTy,
                           IRType ValueTy) override;
  void storeToPointee(IRValue *Slot, IRType AllocTy, IRValue *Val) override;
  VarAlloc allocPointer(const std::string &Name, IRType ElemTy,
                        unsigned AddrSpace = 0) override;
  VarAlloc allocArray(const std::string &Name, AddressSpace AS, IRType ElemTy,
                      size_t NElem) override;

#if defined(PROTEUS_ENABLE_CUDA) || defined(PROTEUS_ENABLE_HIP)
  void setLaunchBoundsForKernel(IRFunction *F, int MaxThreadsPerBlock,
                                int MinBlocksPerSM) override;
#endif

private:
  struct Impl;
  std::unique_ptr<Impl> PImpl;
  TargetModelType TargetModel;
};

} // namespace proteus

#endif // PROTEUS_FRONTEND_MLIR_CODE_BUILDER_H
