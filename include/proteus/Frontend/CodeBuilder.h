#ifndef PROTEUS_FRONTEND_CODE_BUILDER_H
#define PROTEUS_FRONTEND_CODE_BUILDER_H

#include "proteus/AddressSpace.h"
#include "proteus/Error.h"
#include "proteus/Frontend/IRFunction.h"
#include "proteus/Frontend/IRType.h"
#include "proteus/Frontend/IRValue.h"
#include "proteus/Frontend/TargetModel.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace proteus {

/// Hints that control loop transformation metadata emitted by the code
/// builder. Extensible for future hints (tiling, vectorization, etc.).
struct LoopHints {
  bool Unroll = false;
  std::optional<int> UnrollCount;
};

/// Lightweight aggregate returned by the alloc* factory methods.  Each
/// Var specialization stores these fields directly instead of a
/// heap-allocated storage objects.
struct VarAlloc {
  /// The alloca (or global base pointer for arrays).
  IRValue *Slot;
  /// Logical element type (pointee/element for pointer/array).
  IRType ValueTy;
  /// Type of the alloca itself.
  IRType AllocTy;
  /// Address space (relevant for pointers/arrays).
  unsigned AddrSpace = 0;
};

enum class ScopeKind { FUNCTION, IF, FOR, WHILE };

/// Semantic arithmetic operation selector.
enum class ArithOp { Add, Sub, Mul, Div, Rem };

/// Semantic comparison operation selector.
enum class CmpOp { EQ, NE, LT, LE, GT, GE };

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

/// Identifies the concrete backend implementation, enabling static_cast
/// dispatch without RTTI.
enum class CodeBuilderKind { LLVM, MLIR };

/// Abstract code-builder interface.  The frontend (Var.h, Func.h, LoopNest.h)
/// depends only on this class.
class CodeBuilder {
public:
  virtual ~CodeBuilder() = default;

  virtual TargetModelType getTargetModel() const = 0;
  virtual CodeBuilderKind getBackendKind() const = 0;

  // -----------------------------------------------------------------------
  // Function management.
  // -----------------------------------------------------------------------

  /// Create a function with the given name and signature. Kernel intent is
  /// explicit so backends can select the correct IR container up front.
  /// Returns an opaque IRFunction handle owned by this builder.
  virtual IRFunction *addFunction(const std::string &Name, IRType RetTy,
                                  const std::vector<IRType> &ArgTys,
                                  bool IsKernel = false) = 0;

  /// Rename the function identified by \p F.
  virtual void setFunctionName(IRFunction *F, const std::string &Name) = 0;

  /// Return the Nth argument of \p F as an IRValue.
  virtual IRValue *getArg(IRFunction *F, size_t Idx) = 0;

  /// Set \p F as the active function and begin IR emission.
  virtual void beginFunction(IRFunction *F, const char *File, int Line) = 0;
  virtual void endFunction() = 0;

  // -----------------------------------------------------------------------
  // Insertion point management.
  // -----------------------------------------------------------------------
  virtual void setInsertPointAtEntry() = 0;
  virtual void clearInsertPoint() = 0;

  // -----------------------------------------------------------------------
  // Control flow.
  // -----------------------------------------------------------------------
  virtual void beginIf(IRValue *Cond, const char *File, int Line) = 0;
  virtual void endIf() = 0;

  /// IterSlot : alloca holding the loop iterator.
  /// IterTy   : value type of the iterator (must be an integer type).
  /// InitVal  : initial value to store into IterSlot.
  /// UpperBoundVal : exclusive upper bound for the loop condition.
  /// IncVal   : increment added to the iterator on each iteration.
  /// IsSigned : true → ICmpSLT, false → ICmpULT.
  virtual void beginFor(IRValue *IterSlot, IRType IterTy, IRValue *InitVal,
                        IRValue *UpperBoundVal, IRValue *IncVal, bool IsSigned,
                        const char *File, int Line, LoopHints Hints = {}) = 0;
  virtual void endFor() = 0;

  /// CondFn : callable that emits the condition IR at the current insert point
  ///          and returns the resulting i1 Value (true → continue loop).
  virtual void beginWhile(std::function<IRValue *()> CondFn, const char *File,
                          int Line) = 0;
  virtual void endWhile() = 0;

  virtual void createRetVoid() = 0;
  virtual void createRet(IRValue *V) = 0;

  // -----------------------------------------------------------------------
  // Arithmetic.
  // -----------------------------------------------------------------------
  virtual IRValue *createArith(ArithOp Op, IRValue *LHS, IRValue *RHS,
                               IRType Ty) = 0;

  // -----------------------------------------------------------------------
  // Atomics.
  // -----------------------------------------------------------------------
  virtual IRValue *createAtomicAdd(IRValue *Addr, IRValue *Val) = 0;
  virtual IRValue *createAtomicSub(IRValue *Addr, IRValue *Val) = 0;
  virtual IRValue *createAtomicMax(IRValue *Addr, IRValue *Val) = 0;
  virtual IRValue *createAtomicMin(IRValue *Addr, IRValue *Val) = 0;

  // -----------------------------------------------------------------------
  // Comparisons.
  // -----------------------------------------------------------------------
  virtual IRValue *createCmp(CmpOp Op, IRValue *LHS, IRValue *RHS,
                             IRType Ty) = 0;

  // -----------------------------------------------------------------------
  // Logical.
  // -----------------------------------------------------------------------
  virtual IRValue *createAnd(IRValue *LHS, IRValue *RHS) = 0;
  virtual IRValue *createOr(IRValue *LHS, IRValue *RHS) = 0;
  virtual IRValue *createXor(IRValue *LHS, IRValue *RHS) = 0;
  virtual IRValue *createNot(IRValue *Val) = 0;

  // -----------------------------------------------------------------------
  // Load/Store.
  // -----------------------------------------------------------------------
  virtual IRValue *createLoad(IRType Ty, IRValue *Ptr,
                              const std::string &Name = "") = 0;
  virtual void createStore(IRValue *Val, IRValue *Ptr) = 0;

  // -----------------------------------------------------------------------
  // Casts.
  // -----------------------------------------------------------------------
  virtual IRValue *createCast(IRValue *V, IRType FromTy, IRType ToTy) = 0;
  virtual IRValue *createBitCast(IRValue *V, IRType DestTy) = 0;
  virtual IRValue *createZExt(IRValue *V, IRType DestTy) = 0;

  // -----------------------------------------------------------------------
  // Constants.
  // -----------------------------------------------------------------------
  virtual IRValue *getConstantInt(IRType Ty, uint64_t Val) = 0;
  virtual IRValue *getConstantFP(IRType Ty, double Val) = 0;

  // -----------------------------------------------------------------------
  // GEP — semantic element-pointer operations.
  // -----------------------------------------------------------------------
  virtual VarAlloc getElementPtr(IRValue *Base, IRType BaseTy, IRValue *Index,
                                 IRType ElemTy) = 0;
  virtual VarAlloc getElementPtr(IRValue *Base, IRType BaseTy, size_t Index,
                                 IRType ElemTy) = 0;

  // -----------------------------------------------------------------------
  // Calls.
  // -----------------------------------------------------------------------
  virtual IRValue *createCall(const std::string &FName, IRType RetTy,
                              const std::vector<IRType> &ArgTys,
                              const std::vector<IRValue *> &Args) = 0;
  virtual IRValue *createCall(const std::string &FName, IRType RetTy) = 0;

  // -----------------------------------------------------------------------
  // Math intrinsics and GPU builtins.
  // -----------------------------------------------------------------------

  /// Lower a frontend intrinsic name to backend IR.
  virtual IRValue *emitIntrinsic(const std::string &Name, IRType RetTy,
                                 const std::vector<IRValue *> &Args) = 0;

  /// Lower a frontend GPU builtin name to backend IR.
  /// Returns nullptr for void builtins (e.g. syncThreads).
  virtual IRValue *emitBuiltin(const std::string &Name, IRType RetTy,
                               const std::vector<IRValue *> &Args) = 0;

  // -----------------------------------------------------------------------
  // Storage-aware load/store.
  // -----------------------------------------------------------------------

  /// Load the value stored directly in \p Slot (scalar alloca).
  virtual IRValue *loadScalar(IRValue *Slot, IRType ValueTy) = 0;
  /// Store \p Val directly into \p Slot (scalar alloca).
  virtual void storeScalar(IRValue *Slot, IRValue *Val) = 0;
  /// Load the pointer stored in \p Slot (pointer alloca).
  virtual IRValue *loadAddress(IRValue *Slot, IRType AllocTy) = 0;
  /// Store \p Addr into \p Slot (pointer alloca).
  virtual void storeAddress(IRValue *Slot, IRValue *Addr) = 0;
  /// Dereference the pointer stored in \p Slot, then load the pointee.
  virtual IRValue *loadFromPointee(IRValue *Slot, IRType AllocTy,
                                   IRType ValueTy) = 0;
  /// Dereference the pointer stored in \p Slot, then store \p Val to it.
  virtual void storeToPointee(IRValue *Slot, IRType AllocTy, IRValue *Val) = 0;

  // -----------------------------------------------------------------------
  // Alloc factories — return a VarAlloc aggregate.
  // -----------------------------------------------------------------------
  virtual VarAlloc allocScalar(const std::string &Name, IRType ValueTy) = 0;
  virtual VarAlloc allocPointer(const std::string &Name, IRType ElemTy,
                                unsigned AddrSpace = 0) = 0;
  virtual VarAlloc allocArray(const std::string &Name, AddressSpace AS,
                              IRType ElemTy, size_t NElem) = 0;

  // -----------------------------------------------------------------------
  // GPU kernel support (CUDA / HIP only).
  // -----------------------------------------------------------------------
#if defined(PROTEUS_ENABLE_CUDA) || defined(PROTEUS_ENABLE_HIP)
  virtual void setLaunchBoundsForKernel(IRFunction *F, int MaxThreadsPerBlock,
                                        int MinBlocksPerSM) = 0;
#endif
};

} // namespace proteus

#endif // PROTEUS_FRONTEND_CODE_BUILDER_H
