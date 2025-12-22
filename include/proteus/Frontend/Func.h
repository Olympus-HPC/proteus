#ifndef PROTEUS_FRONTEND_FUNC_H
#define PROTEUS_FRONTEND_FUNC_H

#include "proteus/AddressSpace.h"
#include "proteus/Error.h"
#include "proteus/Frontend/Dispatcher.h"
#include "proteus/Frontend/TargetModel.h"
#include "proteus/Frontend/TypeMap.h"
#include "proteus/Frontend/TypeTraits.h"
#include "proteus/Frontend/Var.h"
#include "proteus/Frontend/VarStorage.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace llvm {
class BasicBlock;
class Function;
class FunctionCallee;
class AllocaInst;
class Type;
class Value;
class LLVMContext;
class ArrayType;
} // namespace llvm

namespace proteus {

// NOLINTBEGIN(readability-identifier-naming)
template <typename T>
inline constexpr bool is_mutable_v =
    !std::is_const_v<std::remove_reference_t<T>>;
// NOLINTEND(readability-identifier-naming)

class JitModule;
template <typename T> class LoopBoundInfo;
template <typename T, typename... ForLoopBuilders> class LoopNestBuilder;
template <typename T, typename BodyLambda> class ForLoopBuilder;

// Helper struct to represent the signature of a function.
// Useful to partially-specialize function templates.
template <typename... ArgTs> struct ArgTypeList {};
template <typename T> struct FnSig;
template <typename RetT_, typename... ArgT> struct FnSig<RetT_(ArgT...)> {
  using ArgsTList = ArgTypeList<ArgT...>;
  using RetT = RetT_;
};

using namespace llvm;

struct EmptyLambda {
  void operator()() const {}
};

class FuncBase {
public:
  struct Impl;

  JitModule &getJitModule() { return J; }

  LLVMContext &getContext();

  // Storage creation operations.
  std::unique_ptr<ScalarStorage> createScalarStorage(const std::string &Name,
                                                     Type *AllocaTy);
  std::unique_ptr<PointerStorage>
  createPointerStorage(const std::string &Name, Type *AllocaTy, Type *ElemTy);
  std::unique_ptr<ArrayStorage>
  createArrayStorage(const std::string &Name, AddressSpace AS, Type *ArrTy);

  // Insertion point management.
  void setInsertPoint(BasicBlock *BB);
  void setInsertPointBegin(BasicBlock *BB);
  void setInsertPointAtEntry();
  void clearInsertPoint();
  BasicBlock *getInsertBlock();

  // Basic block management.
  std::tuple<BasicBlock *, BasicBlock *> splitCurrentBlock();
  BasicBlock *createBasicBlock(const std::string &Name = "",
                               BasicBlock *InsertBefore = nullptr);
  void eraseTerminator(BasicBlock *BB);
  BasicBlock *getUniqueSuccessor(BasicBlock *BB);

  // Arithmetic operations.
  Value *createAdd(Value *LHS, Value *RHS);
  Value *createFAdd(Value *LHS, Value *RHS);
  Value *createSub(Value *LHS, Value *RHS);
  Value *createFSub(Value *LHS, Value *RHS);
  Value *createMul(Value *LHS, Value *RHS);
  Value *createFMul(Value *LHS, Value *RHS);
  Value *createUDiv(Value *LHS, Value *RHS);
  Value *createSDiv(Value *LHS, Value *RHS);
  Value *createFDiv(Value *LHS, Value *RHS);
  Value *createURem(Value *LHS, Value *RHS);
  Value *createSRem(Value *LHS, Value *RHS);
  Value *createFRem(Value *LHS, Value *RHS);

  // Logical operations.
  Value *createAnd(Value *LHS, Value *RHS);
  Value *createOr(Value *LHS, Value *RHS);
  Value *createXor(Value *LHS, Value *RHS);
  Value *createNot(Value *Val);

  // Comparison operations.
  Value *createICmpEQ(Value *LHS, Value *RHS);
  Value *createICmpNE(Value *LHS, Value *RHS);
  Value *createICmpSLT(Value *LHS, Value *RHS);
  Value *createICmpSGT(Value *LHS, Value *RHS);
  Value *createICmpSGE(Value *LHS, Value *RHS);
  Value *createICmpSLE(Value *LHS, Value *RHS);
  Value *createICmpUGT(Value *LHS, Value *RHS);
  Value *createICmpUGE(Value *LHS, Value *RHS);
  Value *createICmpULT(Value *LHS, Value *RHS);
  Value *createICmpULE(Value *LHS, Value *RHS);
  Value *createFCmpOEQ(Value *LHS, Value *RHS);
  Value *createFCmpONE(Value *LHS, Value *RHS);
  Value *createFCmpOLT(Value *LHS, Value *RHS);
  Value *createFCmpOLE(Value *LHS, Value *RHS);
  Value *createFCmpOGT(Value *LHS, Value *RHS);
  Value *createFCmpOGE(Value *LHS, Value *RHS);
  Value *createFCmpULT(Value *LHS, Value *RHS);
  Value *createFCmpULE(Value *LHS, Value *RHS);

  // Atomic operations.
  Value *createAtomicAdd(Value *Addr, Value *Val);
  Value *createAtomicSub(Value *Addr, Value *Val);
  Value *createAtomicMax(Value *Addr, Value *Val);
  Value *createAtomicMin(Value *Addr, Value *Val);

  // Load/Store operations.
  Value *createLoad(Type *Ty, Value *Ptr, const std::string &Name = "");
  void createStore(Value *Val, Value *Ptr);

  // Call operations.
  Value *createCall(const std::string &FName, Type *RetTy,
                    const std::vector<Type *> &ArgTys,
                    const std::vector<Value *> &Args);
  Value *createCall(const std::string &FName, Type *RetTy);

  // GEP operations.
  Value *createInBoundsGEP(Type *Ty, Value *Ptr,
                           const std::vector<Value *> IdxList,
                           const std::string &Name = "");
  // NOLINTNEXTLINE
  Value *createConstInBoundsGEP1_64(Type *Ty, Value *Ptr, size_t Idx);
  // NOLINTNEXTLINE
  Value *createConstInBoundsGEP2_64(Type *Ty, Value *Ptr, size_t Idx0,
                                    size_t Idx1);

  // Type operations.
  unsigned getAddressSpace(Type *Ty);
  unsigned getAddressSpaceFromValue(Value *PtrVal);
  Type *getPointerType(Type *ElemTy, unsigned AS);
  Type *getPointerTypeUnqual(Type *ElemTy);
  Type *getInt32Ty();
  Type *getInt16Ty();
  Type *getInt64Ty();
  Type *getFloatTy();
  bool isIntegerTy(Type *Ty);
  bool isFloatingPointTy(Type *Ty);

  // Conversion operations.
  template <typename From, typename To> Value *convert(Value *V) {
    static_assert(std::is_arithmetic_v<From>, "From type must be arithmetic");
    static_assert(std::is_arithmetic_v<To>, "To type must be arithmetic");

    auto &Ctx = getContext();

    if constexpr (std::is_same_v<From, To>) {
      return V;
    }

    Type *DestTy = TypeMap<To>::get(Ctx);

    if constexpr (std::is_integral_v<From> && std::is_floating_point_v<To>) {
      if constexpr (std::is_signed_v<From>) {
        return createSIToFP(V, DestTy);
      }

      return createUIToFP(V, DestTy);
    }

    if constexpr (std::is_floating_point_v<From> && std::is_integral_v<To>) {
      if constexpr (std::is_signed_v<To>) {
        return createFPToSI(V, DestTy);
      }

      return createFPToUI(V, DestTy);
    }

    if constexpr (std::is_integral_v<From> && std::is_integral_v<To>) {
      // FuncBase::createIntCast handles Trunc/SExt/ZExt logic internally
      return createIntCast(V, DestTy, std::is_signed_v<From>);
    }

    if constexpr (std::is_floating_point_v<From> &&
                  std::is_floating_point_v<To>) {
      // FuncBase::createFPCast handles FPExt/FPTrunc logic internally
      return createFPCast(V, DestTy);
    }

    reportFatalError("Unsupported conversion");
  }

  // Cast operations.
  // These handle SExt/ZExt/Trunc and FPExt/FPTrunc automatically
  llvm::Value *createIntCast(llvm::Value *V, llvm::Type *DestTy, bool IsSigned);
  llvm::Value *createFPCast(llvm::Value *V, llvm::Type *DestTy);

  // Specific conversions
  llvm::Value *createSIToFP(llvm::Value *V, llvm::Type *DestTy);
  llvm::Value *createUIToFP(llvm::Value *V, llvm::Type *DestTy);
  llvm::Value *createFPToSI(llvm::Value *V, llvm::Type *DestTy);
  llvm::Value *createFPToUI(llvm::Value *V, llvm::Type *DestTy);
  llvm::Value *createBitCast(llvm::Value *V, llvm::Type *DestTy);

  // Constant creation.
  llvm::Value *getConstantInt(llvm::Type *Ty, uint64_t Val);
  llvm::Value *getConstantFP(llvm::Type *Ty, double Val);
  Value *createZExt(Value *V, Type *DestTy);

protected:
  JitModule &J;

  std::string Name;
  std::unique_ptr<Impl> PImpl;

  enum class ScopeKind { FUNCTION, IF, FOR, WHILE };

  std::string toString(ScopeKind Kind) {
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

  // Control flow operations.
  void createBr(llvm::BasicBlock *Dest);
  void createCondBr(llvm::Value *Cond, llvm::BasicBlock *True,
                    llvm::BasicBlock *False);
  void createRetVoid();
  void createRet(llvm::Value *V);

  // Scope management.
  void pushScope(const char *File, const int Line, ScopeKind Kind,
                 BasicBlock *NextBlock);

public:
  FuncBase(JitModule &J, const std::string &Name, Type *RetTy,
           const std::vector<Type *> &ArgTys);
  ~FuncBase();

  TargetModelType getTargetModel() const;

  Function *getFunction();
  Value *getArg(size_t Idx);

#if PROTEUS_ENABLE_CUDA || PROTEUS_ENABLE_HIP
  // Kernel management.
  void setKernel();
  void setLaunchBoundsForKernel(int MaxThreadsPerBlock, int MinBlocksPerSM);
#endif

  AllocaInst *emitAlloca(Type *Ty, const std::string &Name,
                         AddressSpace AS = AddressSpace::DEFAULT);

  Value *emitArrayCreate(Type *Ty, AddressSpace AT, const std::string &Name);

  template <typename T> Var<T> declVar(const std::string &Name = "var") {
    static_assert(!std::is_array_v<T>, "Expected non-array type");
    static_assert(!std::is_reference_v<T>,
                  "declVar does not support reference types");

    auto &Ctx = getContext();
    Type *AllocaTy = TypeMap<T>::get(Ctx);

    if constexpr (std::is_pointer_v<T>) {
      Type *PtrElemTy = TypeMap<T>::getPointerElemType(Ctx);
      return Var<T>{createPointerStorage(Name, AllocaTy, PtrElemTy), *this};
    } else {
      return Var<T>{createScalarStorage(Name, AllocaTy), *this};
    }
  }

  template <typename T>
  Var<T> declVar(size_t NElem, AddressSpace AS = AddressSpace::DEFAULT,
                 const std::string &Name = "array_var") {
    static_assert(std::is_array_v<T>, "Expected array type");

    auto *ArrTy = TypeMap<T>::get(getContext(), NElem);

    return Var<T>{createArrayStorage(Name, AS, ArrTy), *this};
  }

  template <typename T>
  Var<T> defVar(const T &Val, const std::string &Name = "var") {
    using RawT = std::remove_const_t<T>;
    Var<RawT> V = declVar<RawT>(Name);
    V = Val;
    return Var<T>(V);
  }

  template <typename T, typename U>
  Var<T> defVar(const Var<U> &Val, const std::string &Name = "var") {
    using RawT = std::remove_const_t<T>;
    Var<RawT> Res = declVar<RawT>(Name);
    Res = Val;
    return Var<T>(Res);
  }

  template <typename T>
  Var<const T> defRuntimeConst(const T &Val,
                               const std::string &Name = "run.const.var") {
    return Var<const T>(defVar<T>(Val, Name));
  }

  template <typename... ArgT> auto defRuntimeConsts(ArgT &&...Args) {
    return std::make_tuple(defRuntimeConst(std::forward<ArgT>(Args))...);
  }

  void beginFunction(const char *File = __builtin_FILE(),
                     int Line = __builtin_LINE());
  void endFunction();

  void beginIf(const Var<bool> &CondVar, const char *File = __builtin_FILE(),
               int Line = __builtin_LINE());
  void endIf();

  template <typename IterT, typename InitT, typename UpperT, typename IncT>
  void beginFor(Var<IterT> &IterVar, const Var<InitT> &InitVar,
                const Var<UpperT> &UpperBound, const Var<IncT> &IncVar,
                const char *File = __builtin_FILE(),
                int Line = __builtin_LINE());
  void endFor();

  template <typename CondLambda>
  void beginWhile(CondLambda &&Cond, const char *File = __builtin_FILE(),
                  int Line = __builtin_LINE());
  void endWhile();

  template <typename Sig>
  std::enable_if_t<!std::is_void_v<typename FnSig<Sig>::RetT>,
                   Var<typename FnSig<Sig>::RetT>>
  call(const std::string &Name);

  template <typename Sig>
  std::enable_if_t<std::is_void_v<typename FnSig<Sig>::RetT>, void>
  call(const std::string &Name);

  template <typename Sig, typename... ArgVars>
  std::enable_if_t<!std::is_void_v<typename FnSig<Sig>::RetT>,
                   Var<typename FnSig<Sig>::RetT>>
  call(const std::string &Name, ArgVars &&...ArgsVars);

  template <typename Sig, typename... ArgVars>
  std::enable_if_t<std::is_void_v<typename FnSig<Sig>::RetT>, void>
  call(const std::string &Name, ArgVars &&...ArgsVars);

  template <typename BuiltinFuncT>
  decltype(auto) callBuiltin(BuiltinFuncT &&BuiltinFunc) {
    using RetT = std::invoke_result_t<BuiltinFuncT &, FuncBase &>;
    if constexpr (std::is_void_v<RetT>) {
      std::invoke(std::forward<BuiltinFuncT>(BuiltinFunc), *this);
    } else {
      return std::invoke(std::forward<BuiltinFuncT>(BuiltinFunc), *this);
    }
  }

  template <typename T>
  std::enable_if_t<is_arithmetic_unref_v<T>, Var<T>>
  atomicAdd(const Var<T *> &Addr, const Var<T> &Val);
  template <typename T>
  std::enable_if_t<is_arithmetic_unref_v<T>, Var<T>>
  atomicSub(const Var<T *> &Addr, const Var<T> &Val);
  template <typename T>
  std::enable_if_t<is_arithmetic_unref_v<T>, Var<T>>
  atomicMax(const Var<T *> &Addr, const Var<T> &Val);
  template <typename T>
  std::enable_if_t<is_arithmetic_unref_v<T>, Var<T>>
  atomicMin(const Var<T *> &Addr, const Var<T> &Val);

  template <typename IterT, typename InitT, typename UpperT, typename IncT,
            typename BodyLambda = EmptyLambda>
  auto forLoop(Var<IterT> &Iter, const Var<InitT> &Init,
               const Var<UpperT> &Upper, const Var<IncT> &Inc,
               BodyLambda &&Body = {}) {
    static_assert(is_mutable_v<IterT>, "Loop iterator must be mutable");
    LoopBoundInfo<IterT> BoundsInfo{Iter, Init, Upper, Inc};
    return ForLoopBuilder<IterT, BodyLambda>(BoundsInfo, *this,
                                             std::forward<BodyLambda>(Body));
  }

  template <typename... LoopBuilders>
  auto buildLoopNest(LoopBuilders &&...Loops) {
    using FirstBuilder = std::remove_reference_t<
        std::tuple_element_t<0, std::tuple<LoopBuilders...>>>;
    using T = typename FirstBuilder::LoopIndexType;
    return LoopNestBuilder<T, std::decay_t<LoopBuilders>...>(
        *this, std::forward<LoopBuilders>(Loops)...);
  }

  template <typename T> void ret(const Var<T> &RetVal);

  void ret();

  const std::string &getName() const { return Name; }

  void setName(const std::string &NewName);

  // Convert the given Var's value to type U and return a new Var holding
  // the converted value.
  template <typename U, typename T>
  std::enable_if_t<std::is_convertible_v<std::remove_reference_t<T>,
                                         std::remove_reference_t<U>>,
                   Var<clean_t<U>>>
  convert(const Var<T> &V) {
    Var<clean_t<U>> Res = declVar<clean_t<U>>("convert.");
    Value *Converted = convert<clean_t<T>, clean_t<U>>(V.loadValue());
    Res.storeValue(Converted);
    return Res;
  }
};

template <typename RetT, typename... ArgT> class Func final : public FuncBase {
private:
  Dispatcher &Dispatch;
  RetT (*CompiledFunc)(ArgT...) = nullptr;
  // Optional because Var<ArgT> is not default constructible.
  std::tuple<std::optional<Var<ArgT>>...> ArgumentsT;

private:
  template <typename T, std::size_t ArgIdx> Var<T> createArg() {
    auto Var = declVar<T>("arg." + std::to_string(ArgIdx));
    if constexpr (std::is_pointer_v<T>) {
      Var.storePointer(FuncBase::getArg(ArgIdx));
    } else {
      Var.storeValue(FuncBase::getArg(ArgIdx));
    }
    return Var;
  }

  template <std::size_t... Is> void declArgsImpl(std::index_sequence<Is...>) {
    setInsertPointAtEntry();

    (std::get<Is>(ArgumentsT).emplace(createArg<ArgT, Is>()), ...);

    clearInsertPoint();
  }

  template <std::size_t... Is> auto getArgsImpl(std::index_sequence<Is...>) {
    return std::tie(*std::get<Is>(ArgumentsT)...);
  }

public:
  Func(JitModule &J, LLVMContext &Ctx, const std::string &Name,
       Dispatcher &Dispatch)
      : FuncBase(J, Name, TypeMap<RetT>::get(Ctx),
                 {TypeMap<ArgT>::get(Ctx)...}),
        Dispatch(Dispatch) {}

  RetT operator()(ArgT... Args);

  void declArgs() { declArgsImpl(std::index_sequence_for<ArgT...>{}); }

  auto getArgs() { return getArgsImpl(std::index_sequence_for<ArgT...>{}); }

  template <std::size_t Idx> auto &getArg() {
    return *std::get<Idx>(ArgumentsT);
  }

  auto getCompiledFunc() const { return CompiledFunc; }

  void setCompiledFunc(RetT (*CompiledFuncIn)(ArgT...)) {
    CompiledFunc = CompiledFuncIn;
  }
};

// beginFor implementation
template <typename IterT, typename InitT, typename UpperT, typename IncT>
void FuncBase::beginFor(Var<IterT> &IterVar, const Var<InitT> &Init,
                        const Var<UpperT> &UpperBound, const Var<IncT> &Inc,
                        const char *File, int Line) {
  static_assert(std::is_integral_v<std::remove_const_t<IterT>>,
                "Loop iterator must be an integral type");
  static_assert(is_mutable_v<IterT>, "Loop iterator must be mutable");

  // Update the terminator of the current basic block due to the split
  // control-flow.
  auto [CurBlock, NextBlock] = splitCurrentBlock();
  pushScope(File, Line, ScopeKind::FOR, NextBlock);

  BasicBlock *Header = createBasicBlock("loop.header", NextBlock);
  BasicBlock *LoopCond = createBasicBlock("loop.cond", NextBlock);
  BasicBlock *Body = createBasicBlock("loop.body", NextBlock);
  BasicBlock *Latch = createBasicBlock("loop.inc", NextBlock);
  BasicBlock *LoopExit = createBasicBlock("loop.end", NextBlock);

  // Erase the old terminator and branch to the header.
  eraseTerminator(CurBlock);
  setInsertPoint(CurBlock);
  { createBr(Header); }

  setInsertPoint(Header);
  {
    IterVar = Init;
    createBr(LoopCond);
  }

  setInsertPoint(LoopCond);
  {
    auto CondVar = IterVar < UpperBound;
    Value *Cond = CondVar.loadValue();
    createCondBr(Cond, Body, LoopExit);
  }

  setInsertPoint(Body);
  createBr(Latch);

  setInsertPoint(Latch);
  {
    IterVar = IterVar + Inc;
    createBr(LoopCond);
  }

  setInsertPoint(LoopExit);
  { createBr(NextBlock); }

  setInsertPointBegin(Body);
}

template <typename CondLambda>
void FuncBase::beginWhile(CondLambda &&Cond, const char *File, int Line) {
  // Update the terminator of the current basic block due to the split
  // control-flow.
  auto [CurBlock, NextBlock] = splitCurrentBlock();
  pushScope(File, Line, ScopeKind::WHILE, NextBlock);

  BasicBlock *LoopCond = createBasicBlock("while.cond", NextBlock);
  BasicBlock *Body = createBasicBlock("while.body", NextBlock);
  BasicBlock *LoopExit = createBasicBlock("while.end", NextBlock);

  eraseTerminator(CurBlock);
  setInsertPoint(CurBlock);
  { createBr(LoopCond); }

  setInsertPoint(LoopCond);
  {
    auto CondVar = Cond();
    Value *CondV = CondVar.loadValue();
    createCondBr(CondV, Body, LoopExit);
  }

  setInsertPoint(Body);
  createBr(LoopCond);

  setInsertPoint(LoopExit);
  { createBr(NextBlock); }

  setInsertPointBegin(Body);
}

// Var implementations (defined here after FuncBase is
// complete) so we have it available.

// Helper function for binary operations on Var types
template <typename T, typename U, typename IntOp, typename FPOp>
Var<std::common_type_t<clean_t<T>, clean_t<U>>>
binOp(const Var<T> &L, const Var<U> &R, IntOp IOp, FPOp FOp) {
  using CommonT = std::common_type_t<clean_t<T>, clean_t<U>>;

  FuncBase &Fn = L.Fn;
  if (&Fn != &R.Fn)
    reportFatalError("Variables should belong to the same function");

  Value *LHS = Fn.convert<clean_t<T>, CommonT>(L.loadValue());
  Value *RHS = Fn.convert<clean_t<U>, CommonT>(R.loadValue());

  Value *Result = nullptr;
  if constexpr (std::is_integral_v<CommonT>) {
    Result = IOp(Fn, LHS, RHS);
  } else {
    Result = FOp(Fn, LHS, RHS);
  }

  auto ResultVar = Fn.declVar<CommonT>("res.");
  ResultVar.storeValue(Result);

  return ResultVar;
}

// Helper function for compound assignment with a constant
template <typename T, typename U, typename IntOp, typename FPOp>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &
compoundAssignConst(Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &LHS,
                    const U &ConstValue, IntOp IOp, FPOp FOp) {
  static_assert(std::is_convertible_v<clean_t<U>, clean_t<T>>,
                "U must be convertible to T");

  auto &Ctx = LHS.Fn.getContext();
  Type *RHSType = TypeMap<clean_t<U>>::get(Ctx);

  Value *RHS = nullptr;
  if constexpr (std::is_integral_v<clean_t<U>>) {
    RHS = LHS.Fn.getConstantInt(RHSType, ConstValue);
  } else {
    RHS = LHS.Fn.getConstantFP(RHSType, ConstValue);
  }

  Value *LHSVal = LHS.loadValue();

  RHS = LHS.Fn.template convert<clean_t<U>, clean_t<T>>(RHS);
  Value *Result = nullptr;

  if constexpr (std::is_integral_v<clean_t<T>>) {
    Result = IOp(LHS.Fn, LHSVal, RHS);
  } else {
    static_assert(std::is_floating_point_v<clean_t<T>>, "Unsupported type");
    Result = FOp(LHS.Fn, LHSVal, RHS);
  }

  LHS.storeValue(Result);
  return LHS;
}

// Helper function for comparison operations on Var types
template <typename T, typename U, typename IntOp, typename FPOp>
Var<bool> cmpOp(const Var<T> &L, const Var<U> &R, IntOp IOp, FPOp FOp) {
  FuncBase &Fn = L.Fn;
  if (&Fn != &R.Fn)
    reportFatalError("Variables should belong to the same function");

  Value *LHS = L.loadValue();
  Value *RHS = Fn.convert<clean_t<U>, clean_t<T>>(R.loadValue());

  Value *Result = nullptr;
  if constexpr (std::is_integral_v<clean_t<T>>) {
    Result = IOp(Fn, LHS, RHS);
  } else {
    static_assert(std::is_floating_point_v<clean_t<T>>, "Unsupported type");
    Result = FOp(Fn, LHS, RHS);
  }

  auto ResultVar = Fn.declVar<bool>("res.");
  ResultVar.storeValue(Result);

  return ResultVar;
}

template <typename T>
template <typename U, typename>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::Var(const Var<U> &V)
    : VarStorageOwner<VarStorage>(V.Fn) {
  // Allocate storage for the target type T.
  Type *TargetTy = TypeMap<clean_t<T>>::get(Fn.getContext());
  Storage = Fn.createScalarStorage("conv.var", TargetTy);

  auto *Converted = Fn.convert<clean_t<U>, clean_t<T>>(V.loadValue());
  storeValue(Converted);
}

template <typename T>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator=(const Var &V) {
  static_assert(is_mutable_v<T>, "Cannot assign to Var<const T>");
  storeValue(V.loadValue());
  return *this;
}

template <typename T>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator=(Var &&V) {
  static_assert(is_mutable_v<T>, "Cannot assign to Var<const T>");
  if (this->Storage == nullptr) {
    // If we don't have storage, clone it from the source.
    Storage = V.Storage->clone();
  } else {
    // If we have storage, copy the value.
    storeValue(V.loadValue());
  }
  return *this;
}

template <typename T>
Var<std::add_pointer_t<T>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::getAddress() {
  if constexpr (std::is_reference_v<T>) {
    // For references, load the pointer to create a T*.
    auto *PtrStorage = static_cast<PointerStorage *>(Storage.get());
    Value *PtrVal = PtrStorage->loadPointer();
    Type *ElemTy = PtrStorage->getValueType();
    unsigned AddrSpace = Fn.getAddressSpaceFromValue(PtrVal);
    Type *PtrTy = Fn.getPointerType(ElemTy, AddrSpace);
    PtrVal = Fn.createBitCast(PtrVal, PtrTy);

    std::unique_ptr<PointerStorage> ResultStorage =
        Fn.createPointerStorage("addr.ref.tmp", PtrTy, ElemTy);
    Fn.createStore(PtrVal, ResultStorage->getSlot());

    return Var<std::add_pointer_t<T>>(std::move(ResultStorage), Fn);
  }

  Value *Slot = getSlot();
  Type *ElemTy = getAllocatedType();

  unsigned AddrSpace = Fn.getAddressSpace(getSlotType());
  Type *PtrTy = Fn.getPointerType(ElemTy, AddrSpace);
  Value *PtrVal = Slot;
  PtrVal = Fn.createBitCast(Slot, PtrTy);

  std::unique_ptr<PointerStorage> ResultStorage =
      Fn.createPointerStorage("addr.tmp", PtrTy, ElemTy);
  Fn.createStore(PtrVal, ResultStorage->getSlot());

  return Var<std::add_pointer_t<T>>(std::move(ResultStorage), Fn);
}

template <typename T>
template <typename U>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator=(
    const Var<U> &V) {
  static_assert(is_mutable_v<T>, "Cannot assign to Var<const T>");
  auto *Converted = Fn.convert<clean_t<U>, clean_t<T>>(V.loadValue());
  storeValue(Converted);
  return *this;
}

template <typename T>
template <typename U>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator=(
    const U &ConstValue) {
  static_assert(is_mutable_v<T>, "Cannot assign to Var<const T>");
  static_assert(std::is_arithmetic_v<U>,
                "Can only assign arithmetic types to Var");

  Type *LHSType = getValueType();

  if (Fn.isIntegerTy(LHSType)) {
    storeValue(Fn.getConstantInt(LHSType, ConstValue));
  } else if (Fn.isFloatingPointTy(LHSType)) {
    storeValue(Fn.getConstantFP(LHSType, ConstValue));
  } else {
    reportFatalError("Unsupported type");
  }

  return *this;
}

template <typename T>
template <typename U>
Var<std::common_type_t<T, U>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator+(
    const Var<U> &Other) const {
  return binOp(
      *this, Other,
      [](FuncBase &Fn, Value *L, Value *R) { return Fn.createAdd(L, R); },
      [](FuncBase &Fn, Value *L, Value *R) { return Fn.createFAdd(L, R); });
}

template <typename T>
template <typename U>
Var<std::common_type_t<T, U>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator-(
    const Var<U> &Other) const {
  return binOp(
      *this, Other,
      [](FuncBase &Fn, Value *L, Value *R) { return Fn.createSub(L, R); },
      [](FuncBase &Fn, Value *L, Value *R) { return Fn.createFSub(L, R); });
}

template <typename T>
template <typename U>
Var<std::common_type_t<T, U>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator*(
    const Var<U> &Other) const {
  return binOp(
      *this, Other,
      [](FuncBase &Fn, Value *L, Value *R) { return Fn.createMul(L, R); },
      [](FuncBase &Fn, Value *L, Value *R) { return Fn.createFMul(L, R); });
}

template <typename T>
template <typename U>
Var<std::common_type_t<T, U>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator/(
    const Var<U> &Other) const {
  return binOp(
      *this, Other,
      [](FuncBase &Fn, Value *L, Value *R) { return Fn.createSDiv(L, R); },
      [](FuncBase &Fn, Value *L, Value *R) { return Fn.createFDiv(L, R); });
}

template <typename T>
template <typename U>
Var<std::common_type_t<T, U>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator%(
    const Var<U> &Other) const {
  return binOp(
      *this, Other,
      [](FuncBase &Fn, Value *L, Value *R) { return Fn.createSRem(L, R); },
      [](FuncBase &Fn, Value *L, Value *R) { return Fn.createFRem(L, R); });
}

// Arithmetic operators with ConstValue
template <typename T>
template <typename U>
std::enable_if_t<std::is_arithmetic_v<U>, Var<std::common_type_t<T, U>>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator+(
    const U &ConstValue) const {
  static_assert(std::is_arithmetic_v<U>,
                "Can only add arithmetic types to Var");
  Var<U> Tmp = Fn.defVar<U>(ConstValue, "tmp.");
  return (*this) + Tmp;
}

template <typename T>
template <typename U>
std::enable_if_t<std::is_arithmetic_v<U>, Var<std::common_type_t<T, U>>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator-(
    const U &ConstValue) const {
  static_assert(std::is_arithmetic_v<U>,
                "Can only subtract arithmetic types from Var");
  Var<U> Tmp = Fn.defVar<U>(ConstValue, "tmp.");
  return (*this) - Tmp;
}

template <typename T>
template <typename U>
std::enable_if_t<std::is_arithmetic_v<U>, Var<std::common_type_t<T, U>>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator*(
    const U &ConstValue) const {
  static_assert(std::is_arithmetic_v<U>,
                "Can only multiply Var by arithmetic types");
  Var<U> Tmp = Fn.defVar<U>(ConstValue, "tmp.");
  return (*this) * Tmp;
}

template <typename T>
template <typename U>
std::enable_if_t<std::is_arithmetic_v<U>, Var<std::common_type_t<T, U>>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator/(
    const U &ConstValue) const {
  static_assert(std::is_arithmetic_v<U>,
                "Can only divide Var by arithmetic types");
  Var<U> Tmp = Fn.defVar<U>(ConstValue, "tmp.");
  return (*this) / Tmp;
}

template <typename T>
template <typename U>
std::enable_if_t<std::is_arithmetic_v<U>, Var<std::common_type_t<T, U>>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator%(
    const U &ConstValue) const {
  static_assert(std::is_arithmetic_v<U>,
                "Can only modulo Var by arithmetic types");
  Var<U> Tmp = Fn.defVar<U>(ConstValue, "tmp.");
  return (*this) % Tmp;
}

// Compound assignment operators for Var
template <typename T>
template <typename U>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator+=(
    const Var<U> &Other) {
  static_assert(is_mutable_v<T>, "Cannot use += on Var<const T>");
  auto Result = (*this) + Other;
  *this = Result;
  return *this;
}

template <typename T>
template <typename U>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator+=(
    const U &ConstValue) {
  static_assert(is_mutable_v<T>, "Cannot use += on Var<const T>");
  static_assert(std::is_arithmetic_v<U>,
                "Can only add arithmetic types to Var");
  return compoundAssignConst(
      *this, ConstValue,
      [](FuncBase &Fn, Value *L, Value *R) { return Fn.createAdd(L, R); },
      [](FuncBase &Fn, Value *L, Value *R) { return Fn.createFAdd(L, R); });
}

template <typename T>
template <typename U>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator-=(
    const Var<U> &Other) {
  static_assert(is_mutable_v<T>, "Cannot use -= on Var<const T>");
  auto Result = (*this) - Other;
  *this = Result;
  return *this;
}

template <typename T>
template <typename U>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator-=(
    const U &ConstValue) {
  static_assert(is_mutable_v<T>, "Cannot use -= on Var<const T>");
  static_assert(std::is_arithmetic_v<U>,
                "Can only subtract arithmetic types from Var");
  return compoundAssignConst(
      *this, ConstValue,
      [](FuncBase &Fn, Value *L, Value *R) { return Fn.createSub(L, R); },
      [](FuncBase &Fn, Value *L, Value *R) { return Fn.createFSub(L, R); });
}

template <typename T>
template <typename U>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator*=(
    const Var<U> &Other) {
  static_assert(is_mutable_v<T>, "Cannot use *= on Var<const T>");
  auto Result = (*this) * Other;
  *this = Result;
  return *this;
}

template <typename T>
template <typename U>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator*=(
    const U &ConstValue) {
  static_assert(is_mutable_v<T>, "Cannot use *= on Var<const T>");
  static_assert(std::is_arithmetic_v<U>,
                "Can only multiply Var by arithmetic types");
  return compoundAssignConst(
      *this, ConstValue,
      [](FuncBase &Fn, Value *L, Value *R) { return Fn.createMul(L, R); },
      [](FuncBase &Fn, Value *L, Value *R) { return Fn.createFMul(L, R); });
}

template <typename T>
template <typename U>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator/=(
    const Var<U> &Other) {
  static_assert(is_mutable_v<T>, "Cannot use /= on Var<const T>");
  auto Result = (*this) / Other;
  *this = Result;
  return *this;
}

template <typename T>
template <typename U>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator/=(
    const U &ConstValue) {
  static_assert(is_mutable_v<T>, "Cannot use /= on Var<const T>");
  static_assert(std::is_arithmetic_v<U>,
                "Can only divide Var by arithmetic types");
  return compoundAssignConst(
      *this, ConstValue,
      [](FuncBase &Fn, Value *L, Value *R) { return Fn.createSDiv(L, R); },
      [](FuncBase &Fn, Value *L, Value *R) { return Fn.createFDiv(L, R); });
}

template <typename T>
template <typename U>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator%=(
    const Var<U> &Other) {
  static_assert(is_mutable_v<T>, "Cannot use %= on Var<const T>");
  auto Result = (*this) % Other;
  *this = Result;
  return *this;
}

template <typename T>
template <typename U>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>> &
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator%=(
    const U &ConstValue) {
  static_assert(is_mutable_v<T>, "Cannot use %= on Var<const T>");
  static_assert(std::is_arithmetic_v<U>,
                "Can only modulo Var by arithmetic types");
  return compoundAssignConst(
      *this, ConstValue,
      [](FuncBase &Fn, Value *L, Value *R) { return Fn.createSRem(L, R); },
      [](FuncBase &Fn, Value *L, Value *R) { return Fn.createFRem(L, R); });
}

template <typename T>
Var<clean_t<T>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator-() const {
  auto MinusOne =
      Fn.defVar<clean_t<T>>(static_cast<clean_t<T>>(-1), "minus_one.");
  return MinusOne * (*this);
}

template <typename T>
Var<bool>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator!() const {
  Value *V = loadValue();
  Value *ResV = nullptr;
  if constexpr (std::is_same_v<clean_t<T>, bool>) {
    ResV = Fn.createNot(V);
  } else if constexpr (std::is_integral_v<clean_t<T>>) {
    Value *Zero = Fn.getConstantInt(getValueType(), 0);
    ResV = Fn.createICmpEQ(V, Zero);
  } else {
    Value *Zero = Fn.getConstantFP(getValueType(), 0.0);
    ResV = Fn.createFCmpOEQ(V, Zero);
  }
  auto Ret = Fn.declVar<bool>("not.");
  Ret.storeValue(ResV);
  return Ret;
}

template <typename T>
Var<std::add_lvalue_reference_t<std::remove_extent_t<T>>>
Var<T, std::enable_if_t<std::is_array_v<T>>>::operator[](size_t Index) {
  auto *ArrayTy = getAllocatedType();
  auto *BasePointer = getSlot();

  // GEP into the array aggregate: [0, Index]
  auto *GEP = Fn.createConstInBoundsGEP2_64(ArrayTy, BasePointer, 0, Index);
  Type *ElemTy = getValueType();
  unsigned AddrSpace = Fn.getAddressSpace(getSlotType());
  Type *ElemPtrTy = Fn.getPointerType(ElemTy, AddrSpace);

  std::unique_ptr<PointerStorage> ResultStorage =
      Fn.createPointerStorage("elem.ptr", ElemPtrTy, ElemTy);
  Fn.createStore(GEP, ResultStorage->getSlot());
  return Var<std::add_lvalue_reference_t<std::remove_extent_t<T>>>(
      std::move(ResultStorage), Fn);
}

template <typename T>
template <typename IdxT>
std::enable_if_t<std::is_integral_v<IdxT>,
                 Var<std::add_lvalue_reference_t<std::remove_extent_t<T>>>>
Var<T, std::enable_if_t<std::is_array_v<T>>>::operator[](
    const Var<IdxT> &Index) {
  auto *ArrayTy = getAllocatedType();
  auto *BasePointer = getSlot();

  Value *IdxVal = Index.loadValue();
  Value *Zero = Fn.getConstantInt(Index.getValueType(), 0);
  auto *GEP = Fn.createInBoundsGEP(ArrayTy, BasePointer, {Zero, IdxVal});
  Type *ElemTy = getValueType();
  unsigned AddrSpace = Fn.getAddressSpace(getSlotType());
  Type *ElemPtrTy = Fn.getPointerType(ElemTy, AddrSpace);

  std::unique_ptr<VarStorage> ResultStorage =
      Fn.createPointerStorage("elem.ptr", ElemPtrTy, ElemTy);
  Fn.createStore(GEP, ResultStorage->getSlot());

  return Var<std::add_lvalue_reference_t<std::remove_extent_t<T>>>(
      std::move(ResultStorage), Fn);
}

template <typename T>
Var<std::add_lvalue_reference_t<
    std::remove_pointer_t<std::remove_reference_t<T>>>>
Var<T, std::enable_if_t<is_pointer_unref_v<T>>>::operator[](size_t Index) {
  auto *PointerElemTy =
      TypeMap<std::remove_pointer_t<std::remove_reference_t<T>>>::get(
          Fn.getContext());
  auto *Ptr = loadPointer();
  auto *GEP = Fn.createConstInBoundsGEP1_64(PointerElemTy, Ptr, Index);
  unsigned AddrSpace = Fn.getAddressSpace(getAllocatedType());
  Type *ElemPtrTy = Fn.getPointerType(PointerElemTy, AddrSpace);

  // Create a pointer storage to hold the LValue for the Array[Index].
  std::unique_ptr<PointerStorage> ResultStorage =
      Fn.createPointerStorage("elem.ptr", ElemPtrTy, PointerElemTy);
  Fn.createStore(GEP, ResultStorage->getSlot());

  return Var<std::add_lvalue_reference_t<
      std::remove_pointer_t<std::remove_reference_t<T>>>>(
      std::move(ResultStorage), Fn);
}

template <typename T>
template <typename IdxT>
std::enable_if_t<std::is_arithmetic_v<IdxT>,
                 Var<std::add_lvalue_reference_t<
                     std::remove_pointer_t<std::remove_reference_t<T>>>>>
Var<T, std::enable_if_t<is_pointer_unref_v<T>>>::operator[](
    const Var<IdxT> &Index) {

  auto *PointeeType =
      TypeMap<std::remove_pointer_t<std::remove_reference_t<T>>>::get(
          Fn.getContext());
  auto *Ptr = loadPointer();
  auto *IdxValue = Index.loadValue();
  auto *GEP = Fn.createInBoundsGEP(PointeeType, Ptr, {IdxValue});
  unsigned AddrSpace = Fn.getAddressSpace(getAllocatedType());
  Type *ElemPtrTy = Fn.getPointerType(PointeeType, AddrSpace);

  // Create a pointer storage to hold the LValue for the Array[Index].
  std::unique_ptr<PointerStorage> ResultStorage =
      Fn.createPointerStorage("elem.ptr", ElemPtrTy, PointeeType);
  Fn.createStore(GEP, ResultStorage->getSlot());

  return Var<std::add_lvalue_reference_t<
      std::remove_pointer_t<std::remove_reference_t<T>>>>(
      std::move(ResultStorage), Fn);
}

// Pointer type operator*
template <typename T>
Var<std::add_lvalue_reference_t<
    std::remove_pointer_t<std::remove_reference_t<T>>>>
Var<T, std::enable_if_t<is_pointer_unref_v<T>>>::operator*() {
  return (*this)[0];
}

template <typename T>
Var<std::add_pointer_t<T>>
Var<T, std::enable_if_t<is_pointer_unref_v<T>>>::getAddress() {
  Value *PtrVal = loadPointer();
  Type *ElemTy = getValueType();

  unsigned AddrSpace = Fn.getAddressSpace(getAllocatedType());
  Type *PointeePtrTy = Fn.getPointerType(ElemTy, AddrSpace);
  Type *TargetPtrTy = Fn.getPointerTypeUnqual(PointeePtrTy);

  PtrVal = Fn.createBitCast(PtrVal, PointeePtrTy);

  std::unique_ptr<PointerStorage> ResultStorage =
      Fn.createPointerStorage("addr.ptr.tmp", TargetPtrTy, PointeePtrTy);
  Fn.createStore(PtrVal, ResultStorage->getSlot());

  return Var<std::add_pointer_t<T>>(std::move(ResultStorage), Fn);
}

template <typename T>
template <typename OffsetT>
std::enable_if_t<std::is_arithmetic_v<OffsetT>,
                 Var<T, std::enable_if_t<is_pointer_unref_v<T>>>>
Var<T, std::enable_if_t<is_pointer_unref_v<T>>>::operator+(
    const Var<OffsetT> &Offset) const {
  auto *OffsetVal = Offset.loadValue();
  auto *IdxVal = Fn.convert<OffsetT, int64_t>(OffsetVal);

  auto *BasePtr = loadPointer();
  auto *ElemTy = getValueType();

  auto *GEP = Fn.createInBoundsGEP(ElemTy, BasePtr, IdxVal, "ptr.add");

  unsigned AddrSpace = Fn.getAddressSpace(getAllocatedType());
  auto *ElemPtrTy = Fn.getPointerType(ElemTy, AddrSpace);

  std::unique_ptr<PointerStorage> ResultStorage =
      Fn.createPointerStorage("ptr.add.tmp", ElemPtrTy, ElemTy);
  ;
  Fn.createStore(GEP, ResultStorage->getSlot());

  return Var<T, std::enable_if_t<is_pointer_unref_v<T>>>(
      std::move(ResultStorage), Fn);
}

template <typename T>
template <typename OffsetT>
std::enable_if_t<std::is_arithmetic_v<OffsetT>,
                 Var<T, std::enable_if_t<is_pointer_unref_v<T>>>>
Var<T, std::enable_if_t<is_pointer_unref_v<T>>>::operator+(
    OffsetT Offset) const {
  auto *IntTy = Fn.getInt64Ty();
  Value *IdxVal = Fn.getConstantInt(IntTy, Offset);

  auto *BasePtr = loadPointer();
  auto *ElemTy = getValueType();

  auto *GEP = Fn.createInBoundsGEP(ElemTy, BasePtr, {IdxVal}, "ptr.add");

  unsigned AddrSpace = Fn.getAddressSpace(getAllocatedType());
  auto *ElemPtrTy = Fn.getPointerType(ElemTy, AddrSpace);

  std::unique_ptr<PointerStorage> ResultStorage =
      Fn.createPointerStorage("ptr.add.tmp", ElemPtrTy, ElemTy);
  Fn.createStore(GEP, ResultStorage->getSlot());

  return Var<T, std::enable_if_t<is_pointer_unref_v<T>>>(
      std::move(ResultStorage), Fn);
}

// Comparison operators for Var
template <typename T>
template <typename U>
std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator>(
    const Var<U> &Other) const {
  return cmpOp(
      *this, Other,
      [](FuncBase &Fn, Value *L, Value *R) { return Fn.createICmpSGT(L, R); },
      [](FuncBase &Fn, Value *L, Value *R) { return Fn.createFCmpOGT(L, R); });
}

template <typename T>
template <typename U>
std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator>=(
    const Var<U> &Other) const {
  return cmpOp(
      *this, Other,
      [](FuncBase &Fn, Value *L, Value *R) { return Fn.createICmpSGE(L, R); },
      [](FuncBase &Fn, Value *L, Value *R) { return Fn.createFCmpOGE(L, R); });
}

template <typename T>
template <typename U>
std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator<(
    const Var<U> &Other) const {
  return cmpOp(
      *this, Other,
      [](FuncBase &Fn, Value *L, Value *R) { return Fn.createICmpSLT(L, R); },
      [](FuncBase &Fn, Value *L, Value *R) { return Fn.createFCmpOLT(L, R); });
}

template <typename T>
template <typename U>
std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator<=(
    const Var<U> &Other) const {
  return cmpOp(
      *this, Other,
      [](FuncBase &Fn, Value *L, Value *R) { return Fn.createICmpSLE(L, R); },
      [](FuncBase &Fn, Value *L, Value *R) { return Fn.createFCmpOLE(L, R); });
}

template <typename T>
template <typename U>
std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator==(
    const Var<U> &Other) const {
  return cmpOp(
      *this, Other,
      [](FuncBase &Fn, Value *L, Value *R) { return Fn.createICmpEQ(L, R); },
      [](FuncBase &Fn, Value *L, Value *R) { return Fn.createFCmpOEQ(L, R); });
}

template <typename T>
template <typename U>
std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator!=(
    const Var<U> &Other) const {
  return cmpOp(
      *this, Other,
      [](FuncBase &Fn, Value *L, Value *R) { return Fn.createICmpNE(L, R); },
      [](FuncBase &Fn, Value *L, Value *R) { return Fn.createFCmpONE(L, R); });
}

template <typename T>
template <typename U>
std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator>(
    const U &ConstValue) const {
  Var<U> Tmp = Fn.defVar<U>(ConstValue, "cmp.");
  return (*this) > Tmp;
}

template <typename T>
template <typename U>
std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator>=(
    const U &ConstValue) const {
  Var<U> Tmp = Fn.defVar<U>(ConstValue, "cmp.");
  return (*this) >= Tmp;
}

template <typename T>
template <typename U>
std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator<(
    const U &ConstValue) const {
  Var<U> Tmp = Fn.defVar<U>(ConstValue, "cmp.");
  return (*this) < Tmp;
}

template <typename T>
template <typename U>
std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator<=(
    const U &ConstValue) const {
  auto Tmp = Fn.defVar<U>(ConstValue, "cmp.");
  return (*this) <= Tmp;
}

template <typename T>
template <typename U>
std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator==(
    const U &ConstValue) const {
  Var<U> Tmp = Fn.defVar<U>(ConstValue, "cmp.");
  return (*this) == Tmp;
}

template <typename T>
template <typename U>
std::enable_if_t<is_arithmetic_unref_v<U>, Var<bool>>
Var<T, std::enable_if_t<is_scalar_arithmetic_v<T>>>::operator!=(
    const U &ConstValue) const {
  auto Tmp = Fn.defVar<U>(ConstValue, "cmp.");
  return (*this) != Tmp;
}

// Non-member arithmetic operators for Var
template <typename T, typename U>
std::enable_if_t<std::is_arithmetic_v<T> && std::is_arithmetic_v<U>,
                 Var<std::common_type_t<T, U>>>
operator+(const T &ConstValue, const Var<U> &V) {
  Var<T> Tmp = V.Fn.template defVar<T>(ConstValue, "tmp.");
  return Tmp + V;
}

template <typename T, typename U>
std::enable_if_t<std::is_arithmetic_v<T> && std::is_arithmetic_v<U>,
                 Var<std::common_type_t<T, U>>>
operator-(const T &ConstValue, const Var<U> &V) {
  using CommonType = std::common_type_t<T, U>;
  Var<CommonType> Tmp = V.Fn.template defVar<CommonType>(ConstValue, "tmp.");
  return Tmp - V;
}

template <typename T, typename U>
std::enable_if_t<std::is_arithmetic_v<T> && std::is_arithmetic_v<U>,
                 Var<std::common_type_t<T, U>>>
operator*(const T &ConstValue, const Var<U> &V) {
  Var<T> Tmp = V.Fn.template defVar<T>(ConstValue, "tmp.");
  return Tmp * V;
}

template <typename T, typename U>
std::enable_if_t<std::is_arithmetic_v<T> && std::is_arithmetic_v<U>,
                 Var<std::common_type_t<T, U>>>
operator/(const T &ConstValue, const Var<U> &V) {
  Var<T> Tmp = V.Fn.template defVar<T>(ConstValue, "tmp.");
  return Tmp / V;
}

template <typename T, typename U>
std::enable_if_t<std::is_arithmetic_v<T> && std::is_arithmetic_v<U>,
                 Var<std::common_type_t<T, U>>>
operator%(const T &ConstValue, const Var<U> &V) {
  Var<T> Tmp = V.Fn.template defVar<T>(ConstValue, "tmp.");
  return Tmp % V;
}

template <typename T>
std::enable_if_t<is_arithmetic_unref_v<T>, Var<T>>
FuncBase::atomicAdd(const Var<T *> &Addr, const Var<T> &Val) {
  static_assert(std::is_arithmetic_v<T>, "atomicAdd requires arithmetic type");

  Value *Result = createAtomicAdd(Addr.loadPointer(), Val.loadValue());
  auto Ret = declVar<T>("atomic.add.res.");
  Ret.storeValue(Result);
  return Ret;
}

template <typename T>
std::enable_if_t<is_arithmetic_unref_v<T>, Var<T>>
FuncBase::atomicSub(const Var<T *> &Addr, const Var<T> &Val) {
  static_assert(std::is_arithmetic_v<T>, "atomicSub requires arithmetic type");

  Value *Result = createAtomicSub(Addr.loadPointer(), Val.loadValue());
  auto Ret = declVar<T>("atomic.sub.res.");
  Ret.storeValue(Result);
  return Ret;
}

template <typename T>
std::enable_if_t<is_arithmetic_unref_v<T>, Var<T>>
FuncBase::atomicMax(const Var<T *> &Addr, const Var<T> &Val) {
  static_assert(std::is_arithmetic_v<T>, "atomicMax requires arithmetic type");

  Value *Result = createAtomicMax(Addr.loadPointer(), Val.loadValue());
  auto Ret = declVar<T>("atomic.max.res.");
  Ret.storeValue(Result);
  return Ret;
}

template <typename T>
std::enable_if_t<is_arithmetic_unref_v<T>, Var<T>>
FuncBase::atomicMin(const Var<T *> &Addr, const Var<T> &Val) {
  static_assert(std::is_arithmetic_v<T>, "atomicMin requires arithmetic type");

  Value *Result = createAtomicMin(Addr.loadPointer(), Val.loadValue());
  auto Ret = declVar<T>("atomic.min.res.");
  Ret.storeValue(Result);
  return Ret;
}

inline void FuncBase::ret() { createRetVoid(); }

template <typename T> void FuncBase::ret(const Var<T> &RetVal) {
  Value *RetValue = RetVal.loadValue();
  createRet(RetValue);
}

// Helper struct to convert Var operands to a target type T.
// Used by emitIntrinsic to convert all operands to the intrinsic's result
// type. C++17 doesn't support template parameters on lambdas, so we use a
// struct.
template <typename T> struct IntrinsicOperandConverter {
  FuncBase &Fn;

  template <typename U> Value *operator()(const Var<U> &Operand) const {
    return Fn.convert<clean_t<U>, clean_t<T>>(Operand.loadValue());
  }
};

// Helper for emitting intrinsics with Var
template <typename T, typename... Operands>
static Var<T> emitIntrinsic(const std::string &IntrinsicName, Type *ResultType,
                            const Operands &...Ops) {
  static_assert(sizeof...(Ops) > 0, "Intrinsic requires at least one operand");

  auto &Fn = std::get<0>(std::tie(Ops...)).Fn;
  auto CheckFn = [&Fn](const auto &Operand) {
    if (&Operand.Fn != &Fn)
      reportFatalError("Variables should belong to the same function");
  };
  (CheckFn(Ops), ...);

  IntrinsicOperandConverter<T> ConvertOperand{Fn};

  // All operands are converted to the result type.
  std::vector<Type *> ArgTys(sizeof...(Ops), ResultType);
  Value *Call = Fn.createCall(IntrinsicName, ResultType, ArgTys,
                              {ConvertOperand(Ops)...});

  auto ResultVar = Fn.template declVar<T>("res.");
  ResultVar.storeValue(Call);
  return ResultVar;
}

// Math intrinsics for Var
template <typename T> Var<float> powf(const Var<float> &L, const Var<T> &R) {
  static_assert(std::is_convertible_v<T, float>,
                "powf requires floating-point type");

  auto *ResultType = R.Fn.getFloatTy();
  auto RFloat = R.Fn.template convert<float>(R);
  std::string IntrinsicName = "llvm.pow.f32";
#if PROTEUS_ENABLE_CUDA
  if (L.Fn.getTargetModel() == TargetModelType::CUDA)
    IntrinsicName = "__nv_powf";
#endif

  return emitIntrinsic<float>(IntrinsicName, ResultType, L, RFloat);
}

template <typename T> Var<float> sqrtf(const Var<T> &R) {
  static_assert(std::is_convertible_v<T, float>,
                "sqrtf requires floating-point type");

  auto *ResultType = R.Fn.getFloatTy();
  auto RFloat = R.Fn.template convert<float>(R);
  std::string IntrinsicName = "llvm.sqrt.f32";
#if PROTEUS_ENABLE_CUDA
  if (R.Fn.getTargetModel() == TargetModelType::CUDA)
    IntrinsicName = "__nv_sqrtf";
#endif

  return emitIntrinsic<float>(IntrinsicName, ResultType, RFloat);
}

template <typename T> Var<float> expf(const Var<T> &R) {
  static_assert(std::is_convertible_v<T, float>,
                "expf requires floating-point type");

  auto *ResultType = R.Fn.getFloatTy();
  auto RFloat = R.Fn.template convert<float>(R);
  std::string IntrinsicName = "llvm.exp.f32";
#if PROTEUS_ENABLE_CUDA
  if (R.Fn.getTargetModel() == TargetModelType::CUDA)
    IntrinsicName = "__nv_expf";
#endif

  return emitIntrinsic<float>(IntrinsicName, ResultType, RFloat);
}

template <typename T> Var<float> sinf(const Var<T> &R) {
  static_assert(std::is_convertible_v<T, float>,
                "sinf requires floating-point type");

  auto *ResultType = R.Fn.getFloatTy();
  auto RFloat = R.Fn.template convert<float>(R);
  std::string IntrinsicName = "llvm.sin.f32";
#if PROTEUS_ENABLE_CUDA
  if (R.Fn.getTargetModel() == TargetModelType::CUDA)
    IntrinsicName = "__nv_sinf";
#endif

  return emitIntrinsic<float>(IntrinsicName, ResultType, RFloat);
}

template <typename T> Var<float> cosf(const Var<T> &R) {
  static_assert(std::is_convertible_v<T, float>,
                "cosf requires floating-point type");

  auto *ResultType = R.Fn.getFloatTy();
  auto RFloat = R.Fn.template convert<float>(R);
  std::string IntrinsicName = "llvm.cos.f32";
#if PROTEUS_ENABLE_CUDA
  if (R.Fn.getTargetModel() == TargetModelType::CUDA)
    IntrinsicName = "__nv_cosf";
#endif

  return emitIntrinsic<float>(IntrinsicName, ResultType, RFloat);
}

template <typename T> Var<float> fabs(const Var<T> &R) {
  static_assert(std::is_convertible_v<T, float>,
                "fabs requires floating-point type");

  auto *ResultType = R.Fn.getFloatTy();
  auto RFloat = R.Fn.template convert<float>(R);
  std::string IntrinsicName = "llvm.fabs.f32";
#if PROTEUS_ENABLE_CUDA
  if (R.Fn.getTargetModel() == TargetModelType::CUDA)
    IntrinsicName = "__nv_fabsf";
#endif

  return emitIntrinsic<float>(IntrinsicName, ResultType, RFloat);
}

template <typename T> Var<float> truncf(const Var<T> &R) {
  static_assert(std::is_convertible_v<T, float>,
                "truncf requires floating-point type");

  auto *ResultType = R.Fn.getFloatTy();
  auto RFloat = R.Fn.template convert<float>(R);
  std::string IntrinsicName = "llvm.trunc.f32";
#if PROTEUS_ENABLE_CUDA
  if (R.Fn.getTargetModel() == TargetModelType::CUDA)
    IntrinsicName = "__nv_truncf";
#endif

  return emitIntrinsic<float>(IntrinsicName, ResultType, RFloat);
}

template <typename T> Var<float> logf(const Var<T> &R) {
  static_assert(std::is_convertible_v<T, float>,
                "logf requires floating-point type");

  auto *ResultType = R.Fn.getFloatTy();
  auto RFloat = R.Fn.template convert<float>(R);
  std::string IntrinsicName = "llvm.log.f32";
#if PROTEUS_ENABLE_CUDA
  if (R.Fn.getTargetModel() == TargetModelType::CUDA)
    IntrinsicName = "__nv_logf";
#endif

  return emitIntrinsic<float>(IntrinsicName, ResultType, RFloat);
}

template <typename T> Var<float> absf(const Var<T> &R) {
  static_assert(std::is_convertible_v<T, float>,
                "absf requires floating-point type");

  auto *ResultType = R.Fn.getFloatTy();
  auto RFloat = R.Fn.template convert<float>(R);
  std::string IntrinsicName = "llvm.fabs.f32";
#if PROTEUS_ENABLE_CUDA
  if (R.Fn.getTargetModel() == TargetModelType::CUDA)
    IntrinsicName = "__nv_fabsf";
#endif

  return emitIntrinsic<float>(IntrinsicName, ResultType, RFloat);
}

template <typename T>
std::enable_if_t<is_arithmetic_unref_v<T>, Var<clean_t<T>>>
min(const Var<T> &L, const Var<T> &R) {
  FuncBase &Fn = L.Fn;
  if (&Fn != &R.Fn)
    reportFatalError("Variables should belong to the same function");

  auto ResultVar = Fn.declVar<clean_t<T>>("min_res");
  ResultVar = R;
  Fn.beginIf(L < R);
  { ResultVar = L; }
  Fn.endIf();
  return ResultVar;
}

template <typename T>
std::enable_if_t<is_arithmetic_unref_v<T>, Var<clean_t<T>>>
max(const Var<T> &L, const Var<T> &R) {

  FuncBase &Fn = L.Fn;
  if (&Fn != &R.Fn)
    reportFatalError("Variables should belong to the same function");

  auto ResultVar = Fn.declVar<clean_t<T>>("max_res");
  ResultVar = R;
  Fn.beginIf(L > R);
  { ResultVar = L; }
  Fn.endIf();
  return ResultVar;
}

} // namespace proteus

#endif // PROTEUS_FRONTEND_FUNC_H
