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
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>

#include <deque>

namespace proteus {

using namespace mlir;

// ---------------------------------------------------------------------------
// IRType -> MLIR type helper
// ---------------------------------------------------------------------------

static mlir::Type toMLIRScalarType(IRTypeKind Kind, MLIRContext &Ctx) {
  switch (Kind) {
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
  default:
    reportFatalError("Unsupported scalar IRTypeKind");
  }
}

static mlir::Type toMLIRType(IRType Ty, MLIRContext &Ctx) {
  switch (Ty.Kind) {
  case IRTypeKind::Void:
    return NoneType::get(&Ctx);
  case IRTypeKind::Pointer: {
    mlir::Type ElemTy = toMLIRScalarType(Ty.ElemKind, Ctx);
    return MemRefType::get({ShapedType::kDynamic}, ElemTy);
  }
  case IRTypeKind::Array: {
    mlir::Type ElemTy = toMLIRScalarType(Ty.ElemKind, Ctx);
    return MemRefType::get({static_cast<int64_t>(Ty.NElem)}, ElemTy);
  }
  default:
    return toMLIRScalarType(Ty.Kind, Ctx);
  }
}

static bool isSupportedAtomicFloatType(mlir::Type Ty) {
  return Ty.isF32() || Ty.isF64();
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

  // Side table for pointer (base memref, offset slot) representation.
  struct PointerInfo {
    mlir::Value BaseMemRef;
  };
  llvm::DenseMap<mlir::Value, PointerInfo> PointerMap;

  enum class AtomicOp { Add, Sub, Max, Min };

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

  std::pair<mlir::Value, mlir::Value> resolveAtomicAddress(IRValue *Addr) {
    Value Slot = unwrap(Addr);
    auto It = PointerMap.find(Slot);
    if (It == PointerMap.end())
      reportFatalError("atomic on non-pointer address");

    Value Base = It->second.BaseMemRef;
    if (!Base)
      reportFatalError("atomic on non-pointer address");

    auto SlotTy = dyn_cast<MemRefType>(Slot.getType());
    if (!SlotTy || SlotTy.getRank() != 1 || SlotTy.getShape()[0] != 1 ||
        !SlotTy.getElementType().isIndex())
      reportFatalError("atomic on non-pointer address");

    auto Loc = Builder.getUnknownLoc();
    Value Zero = Builder.create<arith::ConstantIndexOp>(Loc, 0);
    Value Idx = Builder.create<memref::LoadOp>(Loc, Slot, ValueRange{Zero});
    return {Base, Idx};
  }

  std::pair<mlir::Value, mlir::Value> resolvePointerValue(IRValue *Ptr) {
    // Pointer values in the MLIR backend are represented as:
    //   - slot: memref<1xindex> holding the current offset at [0]
    //   - PointerMap[slot]: side-table entry holding the base memref
    // This helper validates the representation and returns {slot, base}.
    Value Slot = unwrap(Ptr);
    auto SlotTy = dyn_cast<MemRefType>(Slot.getType());
    if (!SlotTy || SlotTy.getRank() != 1 || SlotTy.getShape()[0] != 1 ||
        !SlotTy.getElementType().isIndex())
      reportFatalError("expected pointer slot with PointerInfo");

    auto It = PointerMap.find(Slot);
    if (It == PointerMap.end() || !It->second.BaseMemRef)
      reportFatalError("expected pointer slot with PointerInfo");

    return {Slot, It->second.BaseMemRef};
  }

  // Returns true if `v` is a pointer slot tracked in PointerMap.
  bool isPointerValue(mlir::Value V) const { return PointerMap.contains(V); }

  // For pointer values, resolve to {baseMemref, idx} where idx = load(slot[0]).
  std::pair<mlir::Value, mlir::Value>
  resolvePointerAddress(mlir::Value PtrSlot) {
    auto It = PointerMap.find(PtrSlot);
    if (It == PointerMap.end() || !It->second.BaseMemRef)
      reportFatalError(
          "MLIRCodeBuilder::resolvePointerAddress: unknown pointer slot");

    auto SlotTy = dyn_cast<MemRefType>(PtrSlot.getType());
    if (!SlotTy || SlotTy.getRank() != 1 || SlotTy.getShape()[0] != 1 ||
        !SlotTy.getElementType().isIndex())
      reportFatalError(
          "MLIRCodeBuilder::resolvePointerAddress: expected memref<1xindex> "
          "slot");

    auto Loc = Builder.getUnknownLoc();
    Value Zero = Builder.create<arith::ConstantIndexOp>(Loc, 0);
    Value Idx = Builder.create<memref::LoadOp>(Loc, PtrSlot, ValueRange{Zero});
    return {It->second.BaseMemRef, Idx};
  }

  // Returns true for scalar mutable-variable slots of form memref<1xElem>.
  static bool isScalarSlotType(mlir::Type Ty) {
    auto MemRefTy = dyn_cast<MemRefType>(Ty);
    return MemRefTy && MemRefTy.getRank() == 1 && MemRefTy.getShape()[0] == 1;
  }

  mlir::Value emitAtomicRmw(AtomicOp Op, mlir::Value Base, mlir::Value Idx,
                            mlir::Value Val) {
    auto Loc = Builder.getUnknownLoc();
    auto BaseTy = dyn_cast<MemRefType>(Base.getType());
    if (!BaseTy)
      reportFatalError("atomic op not supported for this type");

    Type ElemTy = BaseTy.getElementType();
    if (Val.getType() != ElemTy)
      reportFatalError("atomic op not supported for this type");

    const bool IsInt = mlir::isa<IntegerType>(ElemTy);
    const bool IsFloat = mlir::isa<FloatType>(ElemTy);
    if (!IsInt && !IsFloat)
      reportFatalError("atomic op not supported for this type");
    if (IsFloat && !isSupportedAtomicFloatType(ElemTy))
      reportFatalError("atomic op not supported for this type");

    const char *KindStr = nullptr;
    if (mlir::isa<IntegerType>(ElemTy)) {
      switch (Op) {
      case AtomicOp::Add:
        KindStr = "addi";
        break;
      case AtomicOp::Sub:
        KindStr = nullptr;
        break;
      case AtomicOp::Max:
        KindStr = "maxs";
        break;
      case AtomicOp::Min:
        KindStr = "mins";
        break;
      }
    } else if (mlir::isa<FloatType>(ElemTy)) {
      switch (Op) {
      case AtomicOp::Add:
        KindStr = "addf";
        break;
      case AtomicOp::Sub:
        KindStr = "subf";
        break;
      case AtomicOp::Max:
      case AtomicOp::Min:
        KindStr = nullptr;
        break;
      }
    }

    if (KindStr) {
      if (auto MaybeKind = arith::symbolizeAtomicRMWKind(KindStr)) {
        auto Atomic = Builder.create<memref::AtomicRMWOp>(
            Loc, *MaybeKind, Val, Base, ValueRange{Idx});
        return Atomic.getResult();
      }
    }

    auto Generic =
        Builder.create<memref::GenericAtomicRMWOp>(Loc, Base, ValueRange{Idx});
    {
      OpBuilder::InsertionGuard Guard(Builder);
      Region &AtomicBody = Generic.getAtomicBody();
      if (AtomicBody.empty())
        AtomicBody.push_back(new Block());
      Block &Body = AtomicBody.front();
      if (Body.getNumArguments() == 0)
        Body.addArgument(ElemTy, Loc);
      if (!Body.empty() && Body.back().hasTrait<OpTrait::IsTerminator>())
        Body.back().erase();

      Builder.setInsertionPointToEnd(&Body);

      Value Cur = Body.getArgument(0);
      Value New;
      if (IsInt) {
        switch (Op) {
        case AtomicOp::Add:
          New = Builder.create<arith::AddIOp>(Loc, Cur, Val);
          break;
        case AtomicOp::Sub:
          New = Builder.create<arith::SubIOp>(Loc, Cur, Val);
          break;
        case AtomicOp::Max:
          New = Builder.create<arith::MaxSIOp>(Loc, Cur, Val);
          break;
        case AtomicOp::Min:
          New = Builder.create<arith::MinSIOp>(Loc, Cur, Val);
          break;
        }
      } else {
        switch (Op) {
        case AtomicOp::Add:
          New = Builder.create<arith::AddFOp>(Loc, Cur, Val);
          break;
        case AtomicOp::Sub:
          New = Builder.create<arith::SubFOp>(Loc, Cur, Val);
          break;
        case AtomicOp::Max:
          New = Builder.create<arith::MaximumFOp>(Loc, Cur, Val);
          break;
        case AtomicOp::Min:
          New = Builder.create<arith::MinimumFOp>(Loc, Cur, Val);
          break;
        }
      }

      Builder.create<memref::AtomicYieldOp>(Loc, New);
    }

    return Generic.getResult();
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
IRValue *MLIRCodeBuilder::createLoad(IRType Ty, IRValue *Ptr,
                                     const std::string & /*Name*/) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  mlir::Value PtrV = PImpl->unwrap(Ptr);
  mlir::Type ExpectedTy = toMLIRScalarType(Ty.Kind, PImpl->Context);

  // Pointer-value case: Ptr is a pointer slot (memref<1xindex>) tracked in
  // PointerMap, so load dynamic index from slot and dereference base[idx].
  if (PImpl->isPointerValue(PtrV)) {
    auto [Base, Idx] = PImpl->resolvePointerAddress(PtrV);
    Value Val =
        PImpl->Builder.create<memref::LoadOp>(Loc, Base, ValueRange{Idx});
    if (Val.getType() != ExpectedTy)
      reportFatalError("MLIRCodeBuilder::createLoad: type mismatch for pointer "
                       "dereference load");
    return PImpl->wrap(Val);
  }

  // Scalar-slot case: Ptr is a mutable scalar slot represented as
  // memref<1xT>; load slot[0].
  if (Impl::isScalarSlotType(PtrV.getType())) {
    Value Zero = PImpl->Builder.create<arith::ConstantIndexOp>(Loc, 0);
    Value Val =
        PImpl->Builder.create<memref::LoadOp>(Loc, PtrV, ValueRange{Zero});
    if (Val.getType() != ExpectedTy)
      reportFatalError("MLIRCodeBuilder::createLoad: type mismatch for scalar "
                       "slot load");
    return PImpl->wrap(Val);
  }

  reportFatalError("MLIRCodeBuilder::createLoad: unsupported Ptr form");
}
void MLIRCodeBuilder::createStore(IRValue *Val, IRValue *Ptr) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  mlir::Value V = PImpl->unwrap(Val);
  mlir::Value PtrV = PImpl->unwrap(Ptr);

  // Pointer-value case: Ptr is a tracked pointer slot; store through
  // base[idx] where idx is loaded from slot[0].
  if (PImpl->isPointerValue(PtrV)) {
    auto [Base, Idx] = PImpl->resolvePointerAddress(PtrV);
    auto BaseTy = dyn_cast<MemRefType>(Base.getType());
    if (!BaseTy || BaseTy.getElementType() != V.getType())
      reportFatalError("MLIRCodeBuilder::createStore: pointer pointee type "
                       "mismatch");
    PImpl->Builder.create<memref::StoreOp>(Loc, V, Base, ValueRange{Idx});
    return;
  }

  // Scalar-slot case: Ptr is memref<1xT>; store into slot[0].
  if (Impl::isScalarSlotType(PtrV.getType())) {
    auto SlotTy = cast<MemRefType>(PtrV.getType());
    if (SlotTy.getElementType() != V.getType())
      reportFatalError("MLIRCodeBuilder::createStore: scalar slot type "
                       "mismatch");
    Value Zero = PImpl->Builder.create<arith::ConstantIndexOp>(Loc, 0);
    PImpl->Builder.create<memref::StoreOp>(Loc, V, PtrV, ValueRange{Zero});
    return;
  }

  reportFatalError("MLIRCodeBuilder::createStore: unsupported Ptr form");
}
IRValue *MLIRCodeBuilder::createBitCast(IRValue *, IRType) {
  reportFatalError("createBitCast not yet implemented in MLIR backend");
}
IRValue *MLIRCodeBuilder::createZExt(IRValue *, IRType) {
  reportFatalError("createZExt not yet implemented in MLIR backend");
}
VarAlloc MLIRCodeBuilder::getElementPtr(IRValue *Base, IRType /*BaseTy*/,
                                        IRValue *Index, IRType ElemTy) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  Value IdxV = PImpl->unwrap(Index);

  // Cast index to index type if needed.
  if (!isa<IndexType>(IdxV.getType()))
    IdxV = PImpl->Builder.create<arith::IndexCastOp>(
        Loc, PImpl->Builder.getIndexType(), IdxV);

  // Allocate offset slot at entry block.
  auto IdxMemRefTy = MemRefType::get({1}, PImpl->Builder.getIndexType());
  Value OffsetSlot;
  {
    OpBuilder::InsertionGuard Guard(PImpl->Builder);
    PImpl->Builder.setInsertionPointToStart(PImpl->EntryBlock);
    OffsetSlot = PImpl->Builder.create<memref::AllocaOp>(Loc, IdxMemRefTy);
  }

  Value BaseMemRef;
  Value OffsetToStore;
  Value BaseV = PImpl->unwrap(Base);
  if (auto It = PImpl->PointerMap.find(BaseV); It != PImpl->PointerMap.end()) {
    BaseMemRef = It->second.BaseMemRef;
    if (!BaseMemRef)
      reportFatalError("getElementPtr: null pointer base");

    // Compose GEP offsets when Base is itself a pointer slot.
    Value BaseSlotV = BaseV;
    Value Idx0 = PImpl->Builder.create<arith::ConstantIndexOp>(Loc, 0);
    Value BaseOffset =
        PImpl->Builder.create<memref::LoadOp>(Loc, BaseSlotV, ValueRange{Idx0});
    OffsetToStore = PImpl->Builder.create<arith::AddIOp>(Loc, BaseOffset, IdxV);
  } else {
    BaseMemRef = PImpl->unwrap(Base);
    OffsetToStore = IdxV;
  }

  // Store the index as offset.
  Value Idx0 = PImpl->Builder.create<arith::ConstantIndexOp>(Loc, 0);
  PImpl->Builder.create<memref::StoreOp>(Loc, OffsetToStore, OffsetSlot,
                                         ValueRange{Idx0});

  IRValue *SlotIRV = PImpl->wrap(OffsetSlot);

  PImpl->PointerMap[OffsetSlot] = {BaseMemRef};

  IRType AllocTy{IRTypeKind::Pointer, ElemTy.Signed, 0, ElemTy.Kind};
  return {SlotIRV, ElemTy, AllocTy, 0};
}
// NOLINTNEXTLINE
VarAlloc MLIRCodeBuilder::getElementPtr(IRValue *Base, IRType /*BaseTy*/,
                                        size_t Index, IRType ElemTy) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  Value IdxV =
      PImpl->Builder.create<arith::ConstantIndexOp>(Loc, (int64_t)Index);

  // Allocate offset slot at entry block.
  auto IdxMemRefTy = MemRefType::get({1}, PImpl->Builder.getIndexType());
  Value OffsetSlot;
  {
    OpBuilder::InsertionGuard Guard(PImpl->Builder);
    PImpl->Builder.setInsertionPointToStart(PImpl->EntryBlock);
    OffsetSlot = PImpl->Builder.create<memref::AllocaOp>(Loc, IdxMemRefTy);
  }

  Value BaseMemRef;
  Value OffsetToStore;
  Value BaseV = PImpl->unwrap(Base);
  if (auto It = PImpl->PointerMap.find(BaseV); It != PImpl->PointerMap.end()) {
    BaseMemRef = It->second.BaseMemRef;
    if (!BaseMemRef)
      reportFatalError("getElementPtr: null pointer base");

    // Compose GEP offsets when Base is itself a pointer slot.
    Value BaseSlotV = BaseV;
    Value Idx0 = PImpl->Builder.create<arith::ConstantIndexOp>(Loc, 0);
    Value BaseOffset =
        PImpl->Builder.create<memref::LoadOp>(Loc, BaseSlotV, ValueRange{Idx0});
    OffsetToStore = PImpl->Builder.create<arith::AddIOp>(Loc, BaseOffset, IdxV);
  } else {
    BaseMemRef = PImpl->unwrap(Base);
    OffsetToStore = IdxV;
  }

  // Store the index as offset.
  Value Idx0 = PImpl->Builder.create<arith::ConstantIndexOp>(Loc, 0);
  PImpl->Builder.create<memref::StoreOp>(Loc, OffsetToStore, OffsetSlot,
                                         ValueRange{Idx0});

  IRValue *SlotIRV = PImpl->wrap(OffsetSlot);

  PImpl->PointerMap[OffsetSlot] = {BaseMemRef};

  IRType AllocTy{IRTypeKind::Pointer, ElemTy.Signed, 0, ElemTy.Kind};
  return {SlotIRV, ElemTy, AllocTy, 0};
}
IRValue *MLIRCodeBuilder::createCall(const std::string &, IRType,
                                     const std::vector<IRType> &,
                                     const std::vector<IRValue *> &) {
  reportFatalError("createCall not yet implemented in MLIR backend");
}
IRValue *MLIRCodeBuilder::createCall(const std::string &, IRType) {
  reportFatalError("createCall(noarg) not yet implemented in MLIR backend");
}
IRValue *MLIRCodeBuilder::loadAddress(IRValue *Slot, IRType /*AllocTy*/) {
  auto It = PImpl->PointerMap.find(PImpl->unwrap(Slot));
  if (It == PImpl->PointerMap.end())
    reportFatalError("loadAddress: unknown pointer slot");
  if (!It->second.BaseMemRef)
    reportFatalError("loadAddress: null pointer base");
  return Slot;
}

void MLIRCodeBuilder::storeAddress(IRValue *Slot, IRValue *Addr) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  // LHS must be a pointer slot in our canonical representation.
  Value LhsSlot = PImpl->unwrap(Slot);
  auto LhsSlotTy = dyn_cast<MemRefType>(LhsSlot.getType());
  if (!LhsSlotTy || LhsSlotTy.getRank() != 1 || LhsSlotTy.getShape()[0] != 1 ||
      !LhsSlotTy.getElementType().isIndex())
    reportFatalError("storeAddress: expected pointer slot (memref<1xindex>)");

  auto LhsIt = PImpl->PointerMap.find(LhsSlot);
  Value LhsBase =
      (LhsIt != PImpl->PointerMap.end()) ? LhsIt->second.BaseMemRef : Value{};

  Value RhsIdx;
  Value RhsBase;

  Value AddrV = PImpl->unwrap(Addr);
  auto AddrMemRefTy = dyn_cast<MemRefType>(AddrV.getType());
  bool IsPointerSlotTy = AddrMemRefTy && AddrMemRefTy.getRank() == 1 &&
                         AddrMemRefTy.getShape()[0] == 1 &&
                         AddrMemRefTy.getElementType().isIndex();

  Value Zero = PImpl->Builder.create<arith::ConstantIndexOp>(Loc, 0);
  if (IsPointerSlotTy) {
    // Pointer-to-pointer assignment: copy both dynamic offset and base mapping.
    // This is the key path for re-assignment semantics such as p = &B[i].
    auto [RhsSlot, ResolvedBase] = PImpl->resolvePointerValue(Addr);
    RhsBase = ResolvedBase;
    RhsIdx =
        PImpl->Builder.create<memref::LoadOp>(Loc, RhsSlot, ValueRange{Zero});
  } else {
    // Direct base assignment (e.g. scalar.getAddress() / function arg pointer):
    // base is Addr itself and offset is reset to 0.
    RhsBase = AddrV;
    RhsIdx = Zero;
  }

  auto RhsBaseTy = dyn_cast<MemRefType>(RhsBase.getType());
  if (!RhsBaseTy)
    reportFatalError(
        "storeAddress: expected memref base for pointer assignment");
  if (LhsBase) {
    // Preserve typed-pointer semantics: reject base rebinds with mismatched
    // element types once LHS has an established base element type.
    auto LhsBaseTy = dyn_cast<MemRefType>(LhsBase.getType());
    if (!LhsBaseTy)
      reportFatalError("storeAddress: invalid existing pointer base");
    if (LhsBaseTy.getElementType() != RhsBaseTy.getElementType())
      reportFatalError("pointer reassignment with incompatible element types");
  }

  // Update the mutable slot offset and side-table base together so later
  // dereference/atomic operations observe the reassigned pointer value.
  PImpl->Builder.create<memref::StoreOp>(Loc, RhsIdx, LhsSlot,
                                         ValueRange{Zero});
  PImpl->PointerMap[LhsSlot] = {RhsBase};
}

IRValue *MLIRCodeBuilder::createAtomicAdd(IRValue *Addr, IRValue *Val) {
  auto [Base, Idx] = PImpl->resolveAtomicAddress(Addr);
  Value Old =
      PImpl->emitAtomicRmw(Impl::AtomicOp::Add, Base, Idx, PImpl->unwrap(Val));
  return PImpl->wrap(Old);
}

IRValue *MLIRCodeBuilder::createAtomicSub(IRValue *Addr, IRValue *Val) {
  auto [Base, Idx] = PImpl->resolveAtomicAddress(Addr);
  Value Old =
      PImpl->emitAtomicRmw(Impl::AtomicOp::Sub, Base, Idx, PImpl->unwrap(Val));
  return PImpl->wrap(Old);
}

IRValue *MLIRCodeBuilder::createAtomicMax(IRValue *Addr, IRValue *Val) {
  auto [Base, Idx] = PImpl->resolveAtomicAddress(Addr);
  Value Old =
      PImpl->emitAtomicRmw(Impl::AtomicOp::Max, Base, Idx, PImpl->unwrap(Val));
  return PImpl->wrap(Old);
}

IRValue *MLIRCodeBuilder::createAtomicMin(IRValue *Addr, IRValue *Val) {
  auto [Base, Idx] = PImpl->resolveAtomicAddress(Addr);
  Value Old =
      PImpl->emitAtomicRmw(Impl::AtomicOp::Min, Base, Idx, PImpl->unwrap(Val));
  return PImpl->wrap(Old);
}

IRValue *MLIRCodeBuilder::loadFromPointee(IRValue *Slot, IRType /*AllocTy*/,
                                          IRType /*ValueTy*/) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  Value SlotV = PImpl->unwrap(Slot);
  auto It = PImpl->PointerMap.find(SlotV);
  if (It == PImpl->PointerMap.end())
    reportFatalError("loadFromPointee: unknown pointer slot");
  Value Base = It->second.BaseMemRef;
  // Load offset from the memref<1xindex> slot.
  Value IdxZero = PImpl->Builder.create<arith::ConstantIndexOp>(Loc, 0);
  Value Offset =
      PImpl->Builder.create<memref::LoadOp>(Loc, SlotV, ValueRange{IdxZero});
  // Load element from base[offset].
  Value Result =
      PImpl->Builder.create<memref::LoadOp>(Loc, Base, ValueRange{Offset});
  return PImpl->wrap(Result);
}

void MLIRCodeBuilder::storeToPointee(IRValue *Slot, IRType /*AllocTy*/,
                                     IRValue *Val) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  Value SlotV = PImpl->unwrap(Slot);
  auto It = PImpl->PointerMap.find(SlotV);
  if (It == PImpl->PointerMap.end())
    reportFatalError("storeToPointee: unknown pointer slot");
  Value Base = It->second.BaseMemRef;
  Value IdxZero = PImpl->Builder.create<arith::ConstantIndexOp>(Loc, 0);
  Value Offset =
      PImpl->Builder.create<memref::LoadOp>(Loc, SlotV, ValueRange{IdxZero});
  Value V = PImpl->unwrap(Val);
  PImpl->Builder.create<memref::StoreOp>(Loc, V, Base, ValueRange{Offset});
}

VarAlloc MLIRCodeBuilder::allocPointer(const std::string & /*Name*/,
                                       IRType ElemTy, unsigned AddrSpace) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  auto IdxMemRefTy = MemRefType::get({1}, PImpl->Builder.getIndexType());
  Value OffsetSlot;
  {
    OpBuilder::InsertionGuard Guard(PImpl->Builder);
    PImpl->Builder.setInsertionPointToStart(PImpl->EntryBlock);
    OffsetSlot = PImpl->Builder.create<memref::AllocaOp>(Loc, IdxMemRefTy);
  }
  IRValue *SlotIRV = PImpl->wrap(OffsetSlot);
  // Side table entry will be populated by storeAddress.
  PImpl->PointerMap[OffsetSlot] = {mlir::Value{}};
  IRType AllocTy{IRTypeKind::Pointer, ElemTy.Signed, 0, ElemTy.Kind};
  return {SlotIRV, ElemTy, AllocTy, AddrSpace};
}

VarAlloc MLIRCodeBuilder::allocArray(const std::string & /*Name*/,
                                     AddressSpace /*AS*/, IRType ElemTy,
                                     size_t NElem) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  mlir::Type MLIRElemTy = toMLIRScalarType(ElemTy.Kind, PImpl->Context);
  auto ArrMemRefTy = MemRefType::get({static_cast<int64_t>(NElem)}, MLIRElemTy);
  Value Alloca;
  {
    OpBuilder::InsertionGuard Guard(PImpl->Builder);
    PImpl->Builder.setInsertionPointToStart(PImpl->EntryBlock);
    Alloca = PImpl->Builder.create<memref::AllocaOp>(Loc, ArrMemRefTy);
  }
  IRType AllocTy{IRTypeKind::Array, ElemTy.Signed, NElem, ElemTy.Kind};
  return {PImpl->wrap(Alloca), ElemTy, AllocTy, 0};
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
