#include "proteus/Frontend/MLIRCodeBuilder.h"
#include "proteus/Error.h"
#include "proteus/Frontend/IRType.h"
#include "proteus/impl/MLIRIRFunction.h"
#include "proteus/impl/MLIRIRValue.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpDefinition.h>

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>

#include <deque>

namespace proteus {

using namespace mlir;

// ---------------------------------------------------------------------------
// IRType -> MLIR type helper
// ---------------------------------------------------------------------------

static mlir::Type toMLIRType(IRType Ty, MLIRContext &Ctx) {
  switch (Ty.Kind) {
  case IRTypeKind::Void:
    return NoneType::get(&Ctx);
  case IRTypeKind::Int1:
    return IntegerType::get(&Ctx, 1);
  case IRTypeKind::Int16:
    return IntegerType::get(&Ctx, 16);
  case IRTypeKind::Int32:
    return IntegerType::get(&Ctx, 32);
  case IRTypeKind::Int64:
    return IntegerType::get(&Ctx, 64);
  case IRTypeKind::Float:
    return Float32Type::get(&Ctx);
  case IRTypeKind::Double:
    return Float64Type::get(&Ctx);
  case IRTypeKind::Pointer:
  case IRTypeKind::Array:
    reportFatalError("Pointer/Array IRType not yet supported in MLIR backend");
  default:
    reportFatalError("Unsupported IRType for MLIR backend");
  }
}

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------

struct MLIRCodeBuilder::Impl {
  MLIRContext Context;
  OpBuilder Builder;
  ModuleOp Module;

  // Pointer-stable storage for returned handles.
  std::deque<MLIRIRValue> Values;
  std::deque<MLIRIRFunction> Functions;

  // Current function state.
  mlir::func::FuncOp CurrentFunc;
  Block *EntryBlock = nullptr;

  // Scope stack for structured control flow (scf.if / scf.for / scf.while).
  struct ScopeEntry {
    ScopeKind Kind;
    OpBuilder::InsertPoint SavedIP;
  };
  llvm::SmallVector<ScopeEntry> ScopeStack;

  explicit Impl() : Builder(&Context) {
    Context.loadDialect<mlir::func::FuncDialect, arith::ArithDialect,
                        memref::MemRefDialect, scf::SCFDialect>();
    Module = ModuleOp::create(Builder.getUnknownLoc());
  }

  IRValue *wrap(Value V) {
    Values.emplace_back(V);
    return &Values.back();
  }

  Value unwrap(IRValue *V) { return static_cast<MLIRIRValue *>(V)->V; }

  IRFunction *wrapFunction(mlir::func::FuncOp F) {
    Functions.emplace_back(F);
    return &Functions.back();
  }

  mlir::func::FuncOp unwrapFunction(IRFunction *F) {
    return static_cast<MLIRIRFunction *>(F)->F;
  }
};

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

MLIRCodeBuilder::MLIRCodeBuilder(TargetModelType TM)
    : PImpl(std::make_unique<Impl>()), TargetModel(TM) {}

MLIRCodeBuilder::~MLIRCodeBuilder() = default;

// ---------------------------------------------------------------------------
// print
// ---------------------------------------------------------------------------

void MLIRCodeBuilder::print() { PImpl->Module.print(llvm::outs()); }

// ---------------------------------------------------------------------------
// addFunction
// ---------------------------------------------------------------------------

IRFunction *MLIRCodeBuilder::addFunction(const std::string &Name, IRType RetTy,
                                         const std::vector<IRType> &ArgTys) {
  auto &Ctx = PImpl->Context;

  llvm::SmallVector<mlir::Type> MLIRArgTys;
  MLIRArgTys.reserve(ArgTys.size());
  for (const auto &AT : ArgTys)
    MLIRArgTys.push_back(toMLIRType(AT, Ctx));

  llvm::SmallVector<mlir::Type> RetTys;
  if (RetTy.Kind != IRTypeKind::Void)
    RetTys.push_back(toMLIRType(RetTy, Ctx));

  auto FTy = FunctionType::get(&Ctx, MLIRArgTys, RetTys);
  auto Loc = PImpl->Builder.getUnknownLoc();

  // Insert function at end of module.
  PImpl->Builder.setInsertionPointToEnd(PImpl->Module.getBody());
  auto FuncOp = PImpl->Builder.create<mlir::func::FuncOp>(Loc, Name, FTy);

  // Create entry block with arguments matching the function type.
  FuncOp.addEntryBlock();

  // Track the new function immediately so declArgs() (which runs before
  // beginFunction) can call setInsertPointAtEntry() correctly.
  PImpl->CurrentFunc = FuncOp;
  PImpl->EntryBlock = &FuncOp.getBody().front();

  return PImpl->wrapFunction(FuncOp);
}

// ---------------------------------------------------------------------------
// setFunctionName
// ---------------------------------------------------------------------------

void MLIRCodeBuilder::setFunctionName(IRFunction *F, const std::string &Name) {
  PImpl->unwrapFunction(F).setName(Name);
}

// ---------------------------------------------------------------------------
// getArg
// ---------------------------------------------------------------------------

IRValue *MLIRCodeBuilder::getArg(IRFunction *F, size_t Idx) {
  auto FuncOp = PImpl->unwrapFunction(F);
  Value Arg = FuncOp.getBody().front().getArgument(Idx);
  return PImpl->wrap(Arg);
}

// ---------------------------------------------------------------------------
// beginFunction / endFunction
// ---------------------------------------------------------------------------

void MLIRCodeBuilder::beginFunction(IRFunction *F, const char * /*File*/,
                                    int /*Line*/) {
  auto FuncOp = PImpl->unwrapFunction(F);
  PImpl->CurrentFunc = FuncOp;
  PImpl->EntryBlock = &FuncOp.getBody().front();
  PImpl->Builder.setInsertionPointToEnd(PImpl->EntryBlock);
}

void MLIRCodeBuilder::endFunction() {
  // Insert a void return if the current block has no terminator.
  Block *CurBlock = PImpl->Builder.getInsertionBlock();
  if (CurBlock && (CurBlock->empty() ||
                   !CurBlock->back().hasTrait<OpTrait::IsTerminator>())) {
    auto Loc = PImpl->Builder.getUnknownLoc();
    PImpl->Builder.create<mlir::func::ReturnOp>(Loc);
  }
  PImpl->CurrentFunc = {};
  PImpl->EntryBlock = nullptr;
}

// ---------------------------------------------------------------------------
// Insertion point management
// ---------------------------------------------------------------------------

void MLIRCodeBuilder::setInsertPointAtEntry() {
  if (PImpl->EntryBlock)
    // Mirror LLVM's emitAlloca pattern: set IP to the *end* of the entry
    // block so that allocas (which use InsertionGuard to go to the start)
    // and subsequent stores/body ops are ordered correctly.
    PImpl->Builder.setInsertionPointToEnd(PImpl->EntryBlock);
}

void MLIRCodeBuilder::clearInsertPoint() {
  // Move insertion point to module body end as a neutral sentinel.
  PImpl->Builder.setInsertionPointToEnd(PImpl->Module.getBody());
}

// ---------------------------------------------------------------------------
// createRetVoid / createRet
// ---------------------------------------------------------------------------

void MLIRCodeBuilder::createRetVoid() {
  auto Loc = PImpl->Builder.getUnknownLoc();
  PImpl->Builder.create<mlir::func::ReturnOp>(Loc);
}

void MLIRCodeBuilder::createRet(IRValue *V) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  PImpl->Builder.create<mlir::func::ReturnOp>(Loc, PImpl->unwrap(V));
}

// ---------------------------------------------------------------------------
// createArith
// ---------------------------------------------------------------------------

IRValue *MLIRCodeBuilder::createArith(ArithOp Op, IRValue *LHS, IRValue *RHS,
                                      IRType Ty) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  Value L = PImpl->unwrap(LHS);
  Value R = PImpl->unwrap(RHS);
  Value Result;

  if (isIntegerKind(Ty)) {
    switch (Op) {
    case ArithOp::Add:
      Result = PImpl->Builder.create<arith::AddIOp>(Loc, L, R);
      break;
    case ArithOp::Sub:
      Result = PImpl->Builder.create<arith::SubIOp>(Loc, L, R);
      break;
    case ArithOp::Mul:
      Result = PImpl->Builder.create<arith::MulIOp>(Loc, L, R);
      break;
    case ArithOp::Div:
      if (Ty.Signed)
        Result = PImpl->Builder.create<arith::DivSIOp>(Loc, L, R);
      else
        Result = PImpl->Builder.create<arith::DivUIOp>(Loc, L, R);
      break;
    case ArithOp::Rem:
      if (Ty.Signed)
        Result = PImpl->Builder.create<arith::RemSIOp>(Loc, L, R);
      else
        Result = PImpl->Builder.create<arith::RemUIOp>(Loc, L, R);
      break;
    }
  } else if (isFloatingPointKind(Ty)) {
    switch (Op) {
    case ArithOp::Add:
      Result = PImpl->Builder.create<arith::AddFOp>(Loc, L, R);
      break;
    case ArithOp::Sub:
      Result = PImpl->Builder.create<arith::SubFOp>(Loc, L, R);
      break;
    case ArithOp::Mul:
      Result = PImpl->Builder.create<arith::MulFOp>(Loc, L, R);
      break;
    case ArithOp::Div:
      Result = PImpl->Builder.create<arith::DivFOp>(Loc, L, R);
      break;
    case ArithOp::Rem:
      Result = PImpl->Builder.create<arith::RemFOp>(Loc, L, R);
      break;
    }
  } else {
    reportFatalError("createArith: unsupported IRType");
  }

  return PImpl->wrap(Result);
}

// ---------------------------------------------------------------------------
// createCast
// ---------------------------------------------------------------------------

IRValue *MLIRCodeBuilder::createCast(IRValue *V, IRType FromTy, IRType ToTy) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  Value Src = PImpl->unwrap(V);
  mlir::Type DstTy = toMLIRType(ToTy, PImpl->Context);
  Value Result;

  const bool FromInt = isIntegerKind(FromTy);
  const bool ToInt = isIntegerKind(ToTy);
  const bool FromFloat = isFloatingPointKind(FromTy);
  const bool ToFloat = isFloatingPointKind(ToTy);

  if (FromInt && ToFloat) {
    if (FromTy.Signed)
      Result = PImpl->Builder.create<arith::SIToFPOp>(Loc, DstTy, Src);
    else
      Result = PImpl->Builder.create<arith::UIToFPOp>(Loc, DstTy, Src);
  } else if (FromFloat && ToInt) {
    if (ToTy.Signed)
      Result = PImpl->Builder.create<arith::FPToSIOp>(Loc, DstTy, Src);
    else
      Result = PImpl->Builder.create<arith::FPToUIOp>(Loc, DstTy, Src);
  } else if (FromInt && ToInt) {
    const unsigned FromBits = Src.getType().getIntOrFloatBitWidth();
    const unsigned ToBits = DstTy.getIntOrFloatBitWidth();
    if (ToBits < FromBits)
      Result = PImpl->Builder.create<arith::TruncIOp>(Loc, DstTy, Src);
    else if (FromTy.Signed)
      Result = PImpl->Builder.create<arith::ExtSIOp>(Loc, DstTy, Src);
    else
      Result = PImpl->Builder.create<arith::ExtUIOp>(Loc, DstTy, Src);
  } else if (FromFloat && ToFloat) {
    const unsigned FromBits = Src.getType().getIntOrFloatBitWidth();
    const unsigned ToBits = DstTy.getIntOrFloatBitWidth();
    if (ToBits < FromBits)
      Result = PImpl->Builder.create<arith::TruncFOp>(Loc, DstTy, Src);
    else
      Result = PImpl->Builder.create<arith::ExtFOp>(Loc, DstTy, Src);
  } else {
    reportFatalError("createCast: unsupported type combination");
  }

  return PImpl->wrap(Result);
}

// ---------------------------------------------------------------------------
// getConstantInt / getConstantFP
// ---------------------------------------------------------------------------

IRValue *MLIRCodeBuilder::getConstantInt(IRType Ty, uint64_t Val) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  mlir::Type MLIRTy = toMLIRType(Ty, PImpl->Context);
  auto Attr = IntegerAttr::get(MLIRTy, static_cast<int64_t>(Val));
  Value C = PImpl->Builder.create<arith::ConstantOp>(Loc, Attr);
  return PImpl->wrap(C);
}

IRValue *MLIRCodeBuilder::getConstantFP(IRType Ty, double Val) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  mlir::Type MLIRTy = toMLIRType(Ty, PImpl->Context);
  auto FTy = cast<FloatType>(MLIRTy);
  bool LosesInfo;
  llvm::APFloat APVal(Val); // double precision by default
  APVal.convert(FTy.getFloatSemantics(), llvm::APFloat::rmNearestTiesToEven,
                &LosesInfo);
  auto Attr = FloatAttr::get(FTy, APVal);
  Value C = PImpl->Builder.create<arith::ConstantOp>(Loc, Attr);
  return PImpl->wrap(C);
}

// ---------------------------------------------------------------------------
// loadScalar / storeScalar / allocScalar
// ---------------------------------------------------------------------------

IRValue *MLIRCodeBuilder::loadScalar(IRValue *Slot, IRType /*ValueTy*/) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  Value MemRefVal = PImpl->unwrap(Slot);
  Value Idx = PImpl->Builder.create<arith::ConstantIndexOp>(Loc, 0);
  Value Val =
      PImpl->Builder.create<memref::LoadOp>(Loc, MemRefVal, ValueRange{Idx});
  return PImpl->wrap(Val);
}

void MLIRCodeBuilder::storeScalar(IRValue *Slot, IRValue *Val) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  Value MemRefVal = PImpl->unwrap(Slot);
  Value V = PImpl->unwrap(Val);
  Value Idx = PImpl->Builder.create<arith::ConstantIndexOp>(Loc, 0);
  PImpl->Builder.create<memref::StoreOp>(Loc, V, MemRefVal, ValueRange{Idx});
}

VarAlloc MLIRCodeBuilder::allocScalar(const std::string & /*Name*/,
                                      IRType ValueTy) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  mlir::Type ElemTy = toMLIRType(ValueTy, PImpl->Context);
  // Allocate memref<1xT> — mirrors LLVM's alloca pattern.
  auto MemRefTy = MemRefType::get({1}, ElemTy);

  Value Alloca;
  {
    // Always place the alloca at the start of the entry block.
    OpBuilder::InsertionGuard Guard(PImpl->Builder);
    PImpl->Builder.setInsertionPointToStart(PImpl->EntryBlock);
    Alloca = PImpl->Builder.create<memref::AllocaOp>(Loc, MemRefTy);
  }

  IRType AllocTy;
  AllocTy.Kind = IRTypeKind::Pointer;
  AllocTy.ElemKind = ValueTy.Kind;

  return VarAlloc{PImpl->wrap(Alloca), ValueTy, AllocTy, 0};
}

// ---------------------------------------------------------------------------
// Stub implementations — report fatal error when called
// ---------------------------------------------------------------------------

void MLIRCodeBuilder::beginIf(IRValue *Cond, const char * /*File*/,
                              int /*Line*/) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  Value CondV = PImpl->unwrap(Cond);

  // Save the current insertion point for endIf to restore.
  PImpl->ScopeStack.push_back(
      {ScopeKind::IF, PImpl->Builder.saveInsertionPoint()});

  // Create scf.if with no results (mutations via memref side-effects).
  auto IfOp = PImpl->Builder.create<scf::IfOp>(Loc, /*resultTypes=*/TypeRange{},
                                               CondV, /*withElseRegion=*/false);

  // Set insertion point to the start of the then region.
  PImpl->Builder.setInsertionPointToStart(&IfOp.getThenRegion().front());
}

void MLIRCodeBuilder::endIf() {
  // Ensure the then region has a scf.yield terminator.
  Block *CurBlock = PImpl->Builder.getInsertionBlock();
  if (CurBlock->empty() || !CurBlock->back().hasTrait<OpTrait::IsTerminator>())
    PImpl->Builder.create<scf::YieldOp>(PImpl->Builder.getUnknownLoc());

  // Restore insertion point to after the scf.if.
  auto Scope = PImpl->ScopeStack.pop_back_val();
  PImpl->Builder.restoreInsertionPoint(Scope.SavedIP);
}

void MLIRCodeBuilder::beginFor(IRValue *IterSlot, IRType IterTy,
                               IRValue *InitVal, IRValue *UpperBoundVal,
                               IRValue *IncVal, bool /*IsSigned*/,
                               const char * /*File*/, int /*Line*/,
                               LoopHints /*Hints*/) {
  auto Loc = PImpl->Builder.getUnknownLoc();

  // Cast integer bounds to index type.
  Value LB = PImpl->Builder.create<arith::IndexCastOp>(
      Loc, PImpl->Builder.getIndexType(), PImpl->unwrap(InitVal));
  Value UB = PImpl->Builder.create<arith::IndexCastOp>(
      Loc, PImpl->Builder.getIndexType(), PImpl->unwrap(UpperBoundVal));
  Value Step = PImpl->Builder.create<arith::IndexCastOp>(
      Loc, PImpl->Builder.getIndexType(), PImpl->unwrap(IncVal));

  // Save insertion point for endFor.
  PImpl->ScopeStack.push_back(
      {ScopeKind::FOR, PImpl->Builder.saveInsertionPoint()});

  // Create scf.for with no iter_args (mutation via memref side-effects).
  auto ForOp = PImpl->Builder.create<scf::ForOp>(Loc, LB, UB, Step);

  // Set insertion point inside the body.
  PImpl->Builder.setInsertionPointToStart(ForOp.getBody());

  // Cast the index induction variable back to the original integer type
  // and store into IterSlot so user code can read it.
  Value IV = ForOp.getInductionVar();
  mlir::Type IterMLIRTy = toMLIRType(IterTy, PImpl->Context);
  Value TypedIV =
      PImpl->Builder.create<arith::IndexCastOp>(Loc, IterMLIRTy, IV);
  Value Idx = PImpl->Builder.create<arith::ConstantIndexOp>(Loc, 0);
  Value SlotV = PImpl->unwrap(IterSlot);
  PImpl->Builder.create<memref::StoreOp>(Loc, TypedIV, SlotV, ValueRange{Idx});
}

void MLIRCodeBuilder::endFor() {
  // Ensure the scf.for body has a scf.yield terminator.
  Block *CurBlock = PImpl->Builder.getInsertionBlock();
  if (CurBlock->empty() || !CurBlock->back().hasTrait<OpTrait::IsTerminator>())
    PImpl->Builder.create<scf::YieldOp>(PImpl->Builder.getUnknownLoc());

  auto Scope = PImpl->ScopeStack.pop_back_val();
  PImpl->Builder.restoreInsertionPoint(Scope.SavedIP);
}

void MLIRCodeBuilder::beginWhile(std::function<IRValue *()> CondFn,
                                 const char * /*File*/, int /*Line*/) {
  auto Loc = PImpl->Builder.getUnknownLoc();

  // Save insertion point for endWhile.
  PImpl->ScopeStack.push_back(
      {ScopeKind::WHILE, PImpl->Builder.saveInsertionPoint()});

  // Create scf.while with no iter_args and no results.
  auto WhileOp = PImpl->Builder.create<scf::WhileOp>(
      Loc, /*resultTypes=*/TypeRange{}, /*operands=*/ValueRange{});

  // --- Fill the "before" region (condition). ---
  Block *BeforeBlock = PImpl->Builder.createBlock(&WhileOp.getBefore());
  PImpl->Builder.setInsertionPointToEnd(BeforeBlock);

  // Call CondFn to emit the condition IR into the before region.
  IRValue *CondIRV = CondFn();
  Value CondV = PImpl->unwrap(CondIRV);

  // Terminate the before region with scf.condition.
  PImpl->Builder.create<scf::ConditionOp>(Loc, CondV, /*args=*/ValueRange{});

  // --- Prepare the "after" region (body). ---
  Block *AfterBlock = PImpl->Builder.createBlock(&WhileOp.getAfter());
  PImpl->Builder.setInsertionPointToStart(AfterBlock);
  // User body code will emit here between beginWhile/endWhile.
}

void MLIRCodeBuilder::endWhile() {
  // Ensure the after region has a scf.yield terminator.
  Block *CurBlock = PImpl->Builder.getInsertionBlock();
  if (CurBlock->empty() || !CurBlock->back().hasTrait<OpTrait::IsTerminator>())
    PImpl->Builder.create<scf::YieldOp>(PImpl->Builder.getUnknownLoc());

  auto Scope = PImpl->ScopeStack.pop_back_val();
  PImpl->Builder.restoreInsertionPoint(Scope.SavedIP);
}
IRValue *MLIRCodeBuilder::createAtomicAdd(IRValue *, IRValue *) {
  reportFatalError("createAtomicAdd not yet implemented in MLIR backend");
}
IRValue *MLIRCodeBuilder::createAtomicSub(IRValue *, IRValue *) {
  reportFatalError("createAtomicSub not yet implemented in MLIR backend");
}
IRValue *MLIRCodeBuilder::createAtomicMax(IRValue *, IRValue *) {
  reportFatalError("createAtomicMax not yet implemented in MLIR backend");
}
IRValue *MLIRCodeBuilder::createAtomicMin(IRValue *, IRValue *) {
  reportFatalError("createAtomicMin not yet implemented in MLIR backend");
}
IRValue *MLIRCodeBuilder::createCmp(CmpOp Op, IRValue *LHS, IRValue *RHS,
                                    IRType Ty) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  Value L = PImpl->unwrap(LHS);
  Value R = PImpl->unwrap(RHS);
  Value Result;

  if (isFloatingPointKind(Ty)) {
    arith::CmpFPredicate Pred;
    switch (Op) {
    case CmpOp::EQ:
      Pred = arith::CmpFPredicate::OEQ;
      break;
    case CmpOp::NE:
      Pred = arith::CmpFPredicate::ONE;
      break;
    case CmpOp::LT:
      Pred = arith::CmpFPredicate::OLT;
      break;
    case CmpOp::LE:
      Pred = arith::CmpFPredicate::OLE;
      break;
    case CmpOp::GT:
      Pred = arith::CmpFPredicate::OGT;
      break;
    case CmpOp::GE:
      Pred = arith::CmpFPredicate::OGE;
      break;
    default:
      reportFatalError("createCmp: unknown CmpOp");
    }
    Result = PImpl->Builder.create<arith::CmpFOp>(Loc, Pred, L, R);
  } else {
    arith::CmpIPredicate Pred;
    switch (Op) {
    case CmpOp::EQ:
      Pred = arith::CmpIPredicate::eq;
      break;
    case CmpOp::NE:
      Pred = arith::CmpIPredicate::ne;
      break;
    case CmpOp::LT:
      Pred = Ty.Signed ? arith::CmpIPredicate::slt : arith::CmpIPredicate::ult;
      break;
    case CmpOp::LE:
      Pred = Ty.Signed ? arith::CmpIPredicate::sle : arith::CmpIPredicate::ule;
      break;
    case CmpOp::GT:
      Pred = Ty.Signed ? arith::CmpIPredicate::sgt : arith::CmpIPredicate::ugt;
      break;
    case CmpOp::GE:
      Pred = Ty.Signed ? arith::CmpIPredicate::sge : arith::CmpIPredicate::uge;
      break;
    default:
      reportFatalError("createCmp: unknown CmpOp");
    }
    Result = PImpl->Builder.create<arith::CmpIOp>(Loc, Pred, L, R);
  }
  return PImpl->wrap(Result);
}

IRValue *MLIRCodeBuilder::createAnd(IRValue *LHS, IRValue *RHS) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  Value Result = PImpl->Builder.create<arith::AndIOp>(Loc, PImpl->unwrap(LHS),
                                                      PImpl->unwrap(RHS));
  return PImpl->wrap(Result);
}

IRValue *MLIRCodeBuilder::createOr(IRValue *LHS, IRValue *RHS) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  Value Result = PImpl->Builder.create<arith::OrIOp>(Loc, PImpl->unwrap(LHS),
                                                     PImpl->unwrap(RHS));
  return PImpl->wrap(Result);
}

IRValue *MLIRCodeBuilder::createXor(IRValue *LHS, IRValue *RHS) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  Value Result = PImpl->Builder.create<arith::XOrIOp>(Loc, PImpl->unwrap(LHS),
                                                      PImpl->unwrap(RHS));
  return PImpl->wrap(Result);
}

IRValue *MLIRCodeBuilder::createNot(IRValue *Val) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  // arith.xori %val, %true  where %true is arith.constant 1 : i1
  auto TrueAttr = IntegerAttr::get(IntegerType::get(&PImpl->Context, 1), 1);
  Value TrueVal = PImpl->Builder.create<arith::ConstantOp>(Loc, TrueAttr);
  Value Result =
      PImpl->Builder.create<arith::XOrIOp>(Loc, PImpl->unwrap(Val), TrueVal);
  return PImpl->wrap(Result);
}
IRValue *MLIRCodeBuilder::createLoad(IRType, IRValue *, const std::string &) {
  reportFatalError("createLoad not yet implemented in MLIR backend");
}
void MLIRCodeBuilder::createStore(IRValue *, IRValue *) {
  reportFatalError("createStore not yet implemented in MLIR backend");
}
IRValue *MLIRCodeBuilder::createBitCast(IRValue *, IRType) {
  reportFatalError("createBitCast not yet implemented in MLIR backend");
}
IRValue *MLIRCodeBuilder::createZExt(IRValue *, IRType) {
  reportFatalError("createZExt not yet implemented in MLIR backend");
}
VarAlloc MLIRCodeBuilder::getElementPtr(IRValue *, IRType, IRValue *, IRType) {
  reportFatalError("getElementPtr not yet implemented in MLIR backend");
}
// NOLINTNEXTLINE
VarAlloc MLIRCodeBuilder::getElementPtr(IRValue *, IRType, size_t, IRType) {
  reportFatalError("getElementPtr(size_t) not yet implemented in MLIR backend");
}
IRValue *MLIRCodeBuilder::createCall(const std::string &, IRType,
                                     const std::vector<IRType> &,
                                     const std::vector<IRValue *> &) {
  reportFatalError("createCall not yet implemented in MLIR backend");
}
IRValue *MLIRCodeBuilder::createCall(const std::string &, IRType) {
  reportFatalError("createCall(noarg) not yet implemented in MLIR backend");
}
IRValue *MLIRCodeBuilder::loadAddress(IRValue *, IRType) {
  reportFatalError("loadAddress not yet implemented in MLIR backend");
}
void MLIRCodeBuilder::storeAddress(IRValue *, IRValue *) {
  reportFatalError("storeAddress not yet implemented in MLIR backend");
}
IRValue *MLIRCodeBuilder::loadFromPointee(IRValue *, IRType, IRType) {
  reportFatalError("loadFromPointee not yet implemented in MLIR backend");
}
void MLIRCodeBuilder::storeToPointee(IRValue *, IRType, IRValue *) {
  reportFatalError("storeToPointee not yet implemented in MLIR backend");
}
VarAlloc MLIRCodeBuilder::allocPointer(const std::string &, IRType, unsigned) {
  reportFatalError("allocPointer not yet implemented in MLIR backend");
}
VarAlloc MLIRCodeBuilder::allocArray(const std::string &, AddressSpace, IRType,
                                     size_t) {
  reportFatalError("allocArray not yet implemented in MLIR backend");
}

#if defined(PROTEUS_ENABLE_CUDA) || defined(PROTEUS_ENABLE_HIP)
void MLIRCodeBuilder::setKernel(IRFunction *) {
  reportFatalError("setKernel not yet implemented in MLIR backend");
}
void MLIRCodeBuilder::setLaunchBoundsForKernel(IRFunction *, int, int) {
  reportFatalError(
      "setLaunchBoundsForKernel not yet implemented in MLIR backend");
}
#endif

} // namespace proteus
