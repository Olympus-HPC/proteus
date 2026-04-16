#include "proteus/Frontend/MLIRCodeBuilder.h"
#include "proteus/Error.h"
#include "proteus/Frontend/IRType.h"
#include "proteus/impl/Frontend/MLIRLower.h"
#include "proteus/impl/MLIRIRFunction.h"
#include "proteus/impl/MLIRIRValue.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/Index/IR/IndexDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/SymbolTable.h>

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>

#include <deque>

namespace proteus {

using namespace mlir;

template <typename OpTy, typename BuilderTy, typename... Args>
static OpTy createOp(BuilderTy &Builder, mlir::Location Loc,
                     Args &&...ArgsPack) {
#if LLVM_VERSION_MAJOR >= 22
  return OpTy::create(Builder, Loc, std::forward<Args>(ArgsPack)...);
#else
  return Builder.template create<OpTy>(Loc, std::forward<Args>(ArgsPack)...);
#endif
}

// ---------------------------------------------------------------------------
// IRType -> MLIR type helper
// ---------------------------------------------------------------------------

static mlir::Type toMLIRScalarType(IRTypeKind Kind, MLIRContext &Ctx) {
  switch (Kind) {
  case IRTypeKind::Int1:
    return mlir::IntegerType::get(&Ctx, 1);
  case IRTypeKind::Int16:
    return mlir::IntegerType::get(&Ctx, 16);
  case IRTypeKind::Int32:
    return mlir::IntegerType::get(&Ctx, 32);
  case IRTypeKind::Int64:
    return mlir::IntegerType::get(&Ctx, 64);
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
    // Represent pointers as LLVM dialect opaque pointers. This avoids extra
    // pointer-dialect legalization and matches Proteus' C ABI expectations.
    return mlir::LLVM::LLVMPointerType::get(&Ctx);
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
  TargetModelType TargetModel;
  MLIRContext Context;
  OpBuilder Builder;
  ModuleOp Module;
  gpu::GPUModuleOp DeviceModule;

  // Optional device architecture string used during device lowering.
  // HIP: ROCDL chipset (e.g. gfx90a). CUDA: GPU arch (e.g. sm_80), if set.
  std::string DeviceArch;

  // Lower to LLVM lazily when compilation artifacts are requested.
  MLIRLoweringResult LoweredModule;
  bool LoweredToLLVM = false;

  // Pointer-stable storage for returned handles.
  std::deque<MLIRIRValue> Values;
  std::deque<MLIRIRFunction> Functions;

  // Current function state.
  mlir::Operation *CurrentFuncOp = nullptr;
  bool CurrentIsKernel = false;
  Block *EntryBlock = nullptr;

  // Scope stack for structured control flow (scf.if / scf.for / scf.while).
  struct ScopeEntry {
    ScopeKind Kind;
    OpBuilder::InsertPoint SavedIP;
  };
  llvm::SmallVector<ScopeEntry> ScopeStack;

  mlir::Type toScalarMLIRType(IRType Ty) {
    if (Ty.Kind == IRTypeKind::Pointer || Ty.Kind == IRTypeKind::Array ||
        Ty.Kind == IRTypeKind::Void)
      reportFatalError("expected scalar IRType");
    return toMLIRScalarType(Ty.Kind, Context);
  }

  static unsigned getBitWidthOrZero(mlir::Type Ty) {
    if (auto IntTy = mlir::dyn_cast<mlir::IntegerType>(Ty))
      return IntTy.getWidth();
    if (auto FloatTy = mlir::dyn_cast<mlir::FloatType>(Ty))
      return FloatTy.getWidth();
    return 0;
  }

  static bool isScalarIntOrFloat(mlir::Type Ty) {
    return mlir::isa<mlir::IntegerType>(Ty) || mlir::isa<mlir::FloatType>(Ty);
  }

  VarAlloc emitElementPtrSlot(IRValue *Base, mlir::Value IdxI64,
                              IRType ElemTy) {
    auto Loc = Builder.getUnknownLoc();
    auto I64Ty = mlir::IntegerType::get(&Context, 64);
    auto LLVMPointerTy = mlir::LLVM::LLVMPointerType::get(&Context);

    if (!mlir::isa<mlir::IntegerType>(IdxI64.getType()) ||
        mlir::cast<mlir::IntegerType>(IdxI64.getType()).getWidth() != 64) {
      reportFatalError("getElementPtr: expected i64 element index");
    }

    // Convert base to an LLVM pointer.
    mlir::Value BaseV = unwrap(Base);
    mlir::Value BasePtr;
    mlir::LLVM::LLVMPointerType PtrTy;
    if (auto BasePtrTy =
            dyn_cast<mlir::LLVM::LLVMPointerType>(BaseV.getType())) {
      PtrTy = BasePtrTy;
      BasePtr = BaseV;
    } else if (auto BaseMemRefTy = dyn_cast<MemRefType>(BaseV.getType())) {
      unsigned AddrSpace = 0;
      if (auto MemSpaceAttr = dyn_cast_or_null<mlir::IntegerAttr>(
              BaseMemRefTy.getMemorySpace()))
        AddrSpace = static_cast<unsigned>(MemSpaceAttr.getInt());
      PtrTy = mlir::LLVM::LLVMPointerType::get(&Context, AddrSpace);
      mlir::Value BaseAddrAsIndex =
          createOp<memref::ExtractAlignedPointerAsIndexOp>(Builder, Loc, BaseV);
      mlir::Value BaseAddrI64 =
          createOp<arith::IndexCastUIOp>(Builder, Loc, I64Ty, BaseAddrAsIndex);
      BasePtr =
          createOp<mlir::LLVM::IntToPtrOp>(Builder, Loc, PtrTy, BaseAddrI64);
    } else {
      reportFatalError("getElementPtr: expected !llvm.ptr or memref base");
    }

    mlir::Type GepElemTy = (ElemTy.Kind == IRTypeKind::Pointer)
                               ? mlir::Type(LLVMPointerTy)
                               : toMLIRScalarType(ElemTy.Kind, Context);
    mlir::Value ElemPtr = createOp<mlir::LLVM::GEPOp>(
        Builder, Loc, PtrTy, GepElemTy, BasePtr, ValueRange{IdxI64});

    // Materialize as a pointer slot so reference Vars can call load/store.
    auto PtrSlotTy = MemRefType::get({1}, PtrTy);
    mlir::Value PtrSlot;
    {
      OpBuilder::InsertionGuard Guard(Builder);
      Builder.setInsertionPointToStart(EntryBlock);
      PtrSlot = createOp<memref::AllocaOp>(Builder, Loc, PtrSlotTy);
    }
    mlir::Value Zero = createOp<arith::ConstantIndexOp>(Builder, Loc, 0);
    createOp<memref::StoreOp>(Builder, Loc, ElemPtr, PtrSlot, ValueRange{Zero});

    IRType AllocTy{IRTypeKind::Pointer, ElemTy.Signed, 0, ElemTy.Kind};
    return {wrap(PtrSlot), ElemTy, AllocTy, PtrTy.getAddressSpace()};
  }

  explicit Impl(TargetModelType TM) : TargetModel(TM), Builder(&Context) {
    loadMLIRLoweringDialects(Context);
    Module = ModuleOp::create(Builder.getUnknownLoc());
  }

  IRValue *wrap(mlir::Value V) {
    Values.emplace_back(V);
    return &Values.back();
  }

  mlir::Value unwrap(IRValue *V) { return static_cast<MLIRIRValue *>(V)->V; }

  IRFunction *wrapFunction(mlir::Operation *Op, bool IsKernel, IRType RetTy,
                           const std::vector<IRType> &ArgTys) {
    Functions.emplace_back(Op, IsKernel, RetTy, ArgTys);
    return &Functions.back();
  }

  MLIRIRFunction *unwrapFunction(IRFunction *F) {
    return static_cast<MLIRIRFunction *>(F);
  }

  gpu::GPUModuleOp getOrCreateDeviceModule() {
    if (DeviceModule)
      return DeviceModule;

    constexpr llvm::StringLiteral DeviceModuleName = "kernels";
    if (auto Existing =
            Module.lookupSymbol<gpu::GPUModuleOp>(DeviceModuleName)) {
      DeviceModule = Existing;
      return DeviceModule;
    }

    OpBuilder::InsertionGuard Guard(Builder);
    Builder.setInsertionPointToStart(Module.getBody());
    // CUDA/HIP lowering expects device kernels to live under a dedicated
    // gpu.module symbol, mirroring separate device compilation.
    DeviceModule = createOp<gpu::GPUModuleOp>(Builder, Builder.getUnknownLoc(),
                                              DeviceModuleName);
    return DeviceModule;
  }

  bool isRawPointerAbiType(IRType Ty) {
    if (Ty.Kind != IRTypeKind::Pointer)
      return false;
    return true;
  }

  func::FuncOp getOrCreateFunc(StringRef Name, mlir::FunctionType FTy) {
    if (auto Existing = Module.lookupSymbol<func::FuncOp>(Name)) {
      if (Existing.getFunctionType() != FTy)
        reportFatalError("createCall: function type mismatch for " +
                         Name.str());
      return Existing;
    }

    OpBuilder::InsertionGuard Guard(Builder);
    Builder.setInsertionPointToStart(Module.getBody());
    auto NewFunc =
        createOp<func::FuncOp>(Builder, Builder.getUnknownLoc(), Name, FTy);
    NewFunc.setSymVisibilityAttr(Builder.getStringAttr("private"));
    return NewFunc;
  }

  gpu::GPUFuncOp lookupDeviceFunc(StringRef Name) {
    if (!DeviceModule)
      return gpu::GPUFuncOp{};
    return DeviceModule.lookupSymbol<gpu::GPUFuncOp>(Name);
  }

  func::FuncOp lookupDeviceInternalFunc(StringRef Name) {
    if (!DeviceModule)
      return func::FuncOp{};
    return DeviceModule.lookupSymbol<func::FuncOp>(Name);
  }

  // Returns true for scalar mutable-variable slots of form memref<1xElem>.
  static bool isScalarSlotType(mlir::Type Ty) {
    auto MemRefTy = dyn_cast<MemRefType>(Ty);
    return MemRefTy && MemRefTy.getRank() == 1 && MemRefTy.getShape()[0] == 1;
  }
};

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

MLIRCodeBuilder::MLIRCodeBuilder(TargetModelType TM)
    : PImpl(std::make_unique<Impl>(TM)), TargetModel(TM) {}

MLIRCodeBuilder::~MLIRCodeBuilder() = default;

// ---------------------------------------------------------------------------
// print
// ---------------------------------------------------------------------------

void MLIRCodeBuilder::print() { PImpl->Module.print(llvm::outs()); }

void MLIRCodeBuilder::printLLVMIR(llvm::raw_ostream &OS) {
  ensureLoweredToLLVM(/*OptLevel=*/3, /*TargetTriple=*/"", /*Features=*/"");
  if (!PImpl->LoweredModule.Mod)
    reportFatalError("MLIRCodeBuilder::printLLVMIR: LLVM module ownership was "
                     "already transferred");
  PImpl->LoweredModule.Mod->print(OS, nullptr);
}

void MLIRCodeBuilder::ensureLoweredToLLVM(int OptLevel,
                                          const std::string &TargetTriple,
                                          const std::string &Features) {
  if (PImpl->LoweredToLLVM)
    return;

  MLIRLoweringOptions Options;
  Options.TargetModel = TargetModel;
  Options.DeviceArch = PImpl->DeviceArch;
  Options.OptLevel = OptLevel;
  Options.TargetTriple = TargetTriple;
  Options.Features = Features;
  Options.DiagnosticPrefix = "MLIRCodeBuilder";
  PImpl->LoweredModule = lowerMLIRModuleToLLVM(PImpl->Module, Options);
  PImpl->LoweredToLLVM = true;
}

std::unique_ptr<llvm::LLVMContext> MLIRCodeBuilder::takeContext() {
  ensureLoweredToLLVM(/*OptLevel=*/3, /*TargetTriple=*/"", /*Features=*/"");
  return std::move(PImpl->LoweredModule.Ctx);
}

std::unique_ptr<llvm::Module> MLIRCodeBuilder::takeModule() {
  ensureLoweredToLLVM(/*OptLevel=*/3, /*TargetTriple=*/"", /*Features=*/"");
  return std::move(PImpl->LoweredModule.Mod);
}

void MLIRCodeBuilder::setDeviceArch(const std::string &Arch) {
  if (Arch.empty())
    return;

  if (PImpl->LoweredToLLVM) {
    // Allow redundant calls (e.g. J.printLLVMIR() lowers lazily, then compile()
    // calls setDeviceArch again) as long as the value is consistent.
    if (!PImpl->DeviceArch.empty() && PImpl->DeviceArch != Arch)
      reportFatalError("MLIRCodeBuilder::setDeviceArch called after lowering "
                       "with a different arch (existing=" +
                       PImpl->DeviceArch + ", new=" + Arch + ")");
    PImpl->DeviceArch = Arch;
    return;
  }

  PImpl->DeviceArch = Arch;
}

// ---------------------------------------------------------------------------
// addFunction
// ---------------------------------------------------------------------------

IRFunction *MLIRCodeBuilder::addFunction(const std::string &Name, IRType RetTy,
                                         const std::vector<IRType> &ArgTys,
                                         bool IsKernel) {
  auto &Ctx = PImpl->Context;

  const bool IsDeviceMode = (TargetModel == TargetModelType::CUDA ||
                             TargetModel == TargetModelType::HIP);
  if (!IsDeviceMode && IsKernel)
    reportFatalError("MLIRCodeBuilder::addFunction: host target does not "
                     "support kernel functions");

  const bool IsDeviceInternal = IsDeviceMode && !IsKernel;
  if (IsDeviceInternal && RetTy.Kind == IRTypeKind::Pointer)
    reportFatalError("MLIRCodeBuilder::addFunction: pointer return types are "
                     "not yet supported for internal device calls");

  llvm::SmallVector<mlir::Type> MLIRArgTys;
  MLIRArgTys.reserve(ArgTys.size());
  for (const auto &AT : ArgTys)
    MLIRArgTys.push_back(toMLIRType(AT, Ctx));

  llvm::SmallVector<mlir::Type> RetTys;
  if (RetTy.Kind != IRTypeKind::Void)
    RetTys.push_back(toMLIRType(RetTy, Ctx));

  auto FTy = mlir::FunctionType::get(&Ctx, MLIRArgTys, RetTys);
  auto Loc = PImpl->Builder.getUnknownLoc();

  if (IsDeviceMode) {
    auto DeviceModule = PImpl->getOrCreateDeviceModule();
    if (DeviceModule.getBodyRegion().empty())
      DeviceModule.getBodyRegion().push_back(new Block());
    OpBuilder::InsertionGuard Guard(PImpl->Builder);
    Block &DeviceBody = DeviceModule.getBodyRegion().front();
    // gpu.module owns a terminator (gpu.module_end); insert new ops before it
    // to keep the IR structurally valid.
    if (!DeviceBody.empty() &&
        DeviceBody.back().hasTrait<OpTrait::IsTerminator>()) {
      PImpl->Builder.setInsertionPoint(&DeviceBody.back());
    } else {
      PImpl->Builder.setInsertionPointToEnd(&DeviceBody);
    }

    // In CUDA/HIP mode Proteus emits device-side IR under gpu.module;
    // host-side orchestration remains outside MLIR.
    //
    // Kernels are represented as gpu.func (so GPU lowering can recognize them),
    // while internal device helpers use func.func so `func.call` symbol
    // resolution works without requiring a dedicated gpu.call op.
    if (IsKernel) {
      // Keep gpu.func names equal to frontend names so later mangling can map
      // lowered kernel symbols back to frontend names.
      auto KernelOp = createOp<gpu::GPUFuncOp>(PImpl->Builder, Loc, Name, FTy);
      if (KernelOp.getBody().empty())
        KernelOp.addEntryBlock();
      // Keep kernel marker explicit so downstream GPU passes can identify
      // launchable entry points before target-specific lowering.
      KernelOp->setAttr("gpu.kernel", UnitAttr::get(&PImpl->Context));

      PImpl->CurrentFuncOp = KernelOp.getOperation();
      PImpl->CurrentIsKernel = true;
      PImpl->EntryBlock = &KernelOp.getBody().front();
      return PImpl->wrapFunction(KernelOp.getOperation(), /*IsKernel=*/true,
                                 RetTy, ArgTys);
    }

    auto DevFn = createOp<mlir::func::FuncOp>(PImpl->Builder, Loc, Name, FTy);
    DevFn.setSymVisibilityAttr(PImpl->Builder.getStringAttr("private"));
    DevFn.addEntryBlock();

    PImpl->CurrentFuncOp = DevFn.getOperation();
    PImpl->CurrentIsKernel = false;
    PImpl->EntryBlock = &DevFn.getBody().front();
    return PImpl->wrapFunction(DevFn.getOperation(), /*IsKernel=*/false, RetTy,
                               ArgTys);
  }

  // Host mode emits an exported function callable from the C++ dispatcher
  // (Dispatcher::run<Ret(Args...)>).
  PImpl->Builder.setInsertionPointToEnd(PImpl->Module.getBody());
  auto HostOp = createOp<mlir::func::FuncOp>(PImpl->Builder, Loc, Name, FTy);
  HostOp.addEntryBlock();

  PImpl->CurrentFuncOp = HostOp.getOperation();
  PImpl->CurrentIsKernel = false;
  PImpl->EntryBlock = &HostOp.getBody().front();
  return PImpl->wrapFunction(HostOp.getOperation(), /*IsKernel=*/false, RetTy,
                             ArgTys);
}

// ---------------------------------------------------------------------------
// setFunctionName
// ---------------------------------------------------------------------------

void MLIRCodeBuilder::setFunctionName(IRFunction *F, const std::string &Name) {
  auto *MF = PImpl->unwrapFunction(F);
  if (auto Func = dyn_cast<mlir::func::FuncOp>(MF->Op)) {
    Func.setName(Name);
    return;
  }
  if (auto KernelFn = dyn_cast<gpu::GPUFuncOp>(MF->Op)) {
    // Preserve frontend-visible kernel naming through lowering so
    // JitModule::compile can apply the existing hash suffix mangling.
    KernelFn.setName(Name);
    return;
  }
  reportFatalError("MLIRCodeBuilder::setFunctionName: unsupported function op");
}

// ---------------------------------------------------------------------------
// getArg
// ---------------------------------------------------------------------------

IRValue *MLIRCodeBuilder::getArg(IRFunction *F, size_t Idx) {
  auto *MF = PImpl->unwrapFunction(F);
  if (auto Func = dyn_cast<mlir::func::FuncOp>(MF->Op)) {
    return PImpl->wrap(Func.getBody().front().getArgument(Idx));
  }
  if (auto KernelFn = dyn_cast<gpu::GPUFuncOp>(MF->Op)) {
    // In our current design, device-internal helper functions are emitted as
    // `func.func` ops inside `gpu.module` (not `gpu.func`) because `func.call`
    // cannot target `gpu.func`. As a result, any `gpu.func` we see here should
    // be a kernel entry point.
    if (!MF->IsKernel)
      reportFatalError(
          "MLIRCodeBuilder::getArg: unexpected non-kernel gpu.func");
    return PImpl->wrap(KernelFn.getBody().front().getArgument(Idx));
  }
  reportFatalError("MLIRCodeBuilder::getArg: unsupported function op");
}

// ---------------------------------------------------------------------------
// beginFunction / endFunction
// ---------------------------------------------------------------------------

void MLIRCodeBuilder::beginFunction(IRFunction *F, const char * /*File*/,
                                    int /*Line*/) {
  auto *MF = PImpl->unwrapFunction(F);
  PImpl->CurrentFuncOp = MF->Op;
  PImpl->CurrentIsKernel = MF->IsKernel;
  if (auto Func = dyn_cast<mlir::func::FuncOp>(MF->Op)) {
    PImpl->EntryBlock = &Func.getBody().front();
  } else if (auto KernelFn = dyn_cast<gpu::GPUFuncOp>(MF->Op)) {
    PImpl->EntryBlock = &KernelFn.getBody().front();
  } else {
    reportFatalError("MLIRCodeBuilder::beginFunction: unsupported function op");
  }
  PImpl->Builder.setInsertionPointToEnd(PImpl->EntryBlock);
}

void MLIRCodeBuilder::endFunction() {
  // Insert a void return if the current block has no terminator.
  Block *CurBlock = PImpl->Builder.getInsertionBlock();
  if (CurBlock && (CurBlock->empty() ||
                   !CurBlock->back().hasTrait<OpTrait::IsTerminator>())) {
    auto Loc = PImpl->Builder.getUnknownLoc();
    if (PImpl->CurrentIsKernel)
      createOp<gpu::ReturnOp>(PImpl->Builder, Loc);
    else
      createOp<mlir::func::ReturnOp>(PImpl->Builder, Loc);
  }
  PImpl->CurrentFuncOp = nullptr;
  PImpl->CurrentIsKernel = false;
  PImpl->EntryBlock = nullptr;
}

// ---------------------------------------------------------------------------
// Insertion point management
// ---------------------------------------------------------------------------

void MLIRCodeBuilder::setInsertPointAtEntry() {
  if (!PImpl->EntryBlock)
    reportFatalError(
        "MLIRCodeBuilder::setInsertPointAtEntry: entry block missing");

  // Default to end-of-entry insertion; allocas explicitly insert at start.
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
  if (PImpl->CurrentFuncOp && isa<gpu::GPUFuncOp>(PImpl->CurrentFuncOp))
    createOp<gpu::ReturnOp>(PImpl->Builder, Loc);
  else
    createOp<mlir::func::ReturnOp>(PImpl->Builder, Loc);
}

void MLIRCodeBuilder::createRet(IRValue *V) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  if (PImpl->CurrentFuncOp && isa<gpu::GPUFuncOp>(PImpl->CurrentFuncOp)) {
    if (PImpl->CurrentIsKernel)
      reportFatalError("MLIRCodeBuilder::createRet: kernels must return void");
    createOp<gpu::ReturnOp>(PImpl->Builder, Loc, ValueRange{PImpl->unwrap(V)});
    return;
  }
  createOp<mlir::func::ReturnOp>(PImpl->Builder, Loc, PImpl->unwrap(V));
}

// ---------------------------------------------------------------------------
// createArith
// ---------------------------------------------------------------------------

IRValue *MLIRCodeBuilder::createArith(ArithOp Op, IRValue *LHS, IRValue *RHS,
                                      IRType Ty) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  mlir::Value L = PImpl->unwrap(LHS);
  mlir::Value R = PImpl->unwrap(RHS);
  mlir::Value Result;

  if (isIntegerKind(Ty)) {
    switch (Op) {
    case ArithOp::Add:
      Result = createOp<arith::AddIOp>(PImpl->Builder, Loc, L, R);
      break;
    case ArithOp::Sub:
      Result = createOp<arith::SubIOp>(PImpl->Builder, Loc, L, R);
      break;
    case ArithOp::Mul:
      Result = createOp<arith::MulIOp>(PImpl->Builder, Loc, L, R);
      break;
    case ArithOp::Div:
      if (Ty.Signed)
        Result = createOp<arith::DivSIOp>(PImpl->Builder, Loc, L, R);
      else
        Result = createOp<arith::DivUIOp>(PImpl->Builder, Loc, L, R);
      break;
    case ArithOp::Rem:
      if (Ty.Signed)
        Result = createOp<arith::RemSIOp>(PImpl->Builder, Loc, L, R);
      else
        Result = createOp<arith::RemUIOp>(PImpl->Builder, Loc, L, R);
      break;
    }
  } else if (isFloatingPointKind(Ty)) {
    switch (Op) {
    case ArithOp::Add:
      Result = createOp<arith::AddFOp>(PImpl->Builder, Loc, L, R);
      break;
    case ArithOp::Sub:
      Result = createOp<arith::SubFOp>(PImpl->Builder, Loc, L, R);
      break;
    case ArithOp::Mul:
      Result = createOp<arith::MulFOp>(PImpl->Builder, Loc, L, R);
      break;
    case ArithOp::Div:
      Result = createOp<arith::DivFOp>(PImpl->Builder, Loc, L, R);
      break;
    case ArithOp::Rem:
      Result = createOp<arith::RemFOp>(PImpl->Builder, Loc, L, R);
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
  mlir::Value Src = PImpl->unwrap(V);
  mlir::Type DstTy = toMLIRType(ToTy, PImpl->Context);
  mlir::Value Result;

  const bool FromInt = isIntegerKind(FromTy);
  const bool ToInt = isIntegerKind(ToTy);
  const bool FromFloat = isFloatingPointKind(FromTy);
  const bool ToFloat = isFloatingPointKind(ToTy);

  if (FromInt && ToFloat) {
    if (FromTy.Signed)
      Result = createOp<arith::SIToFPOp>(PImpl->Builder, Loc, DstTy, Src);
    else
      Result = createOp<arith::UIToFPOp>(PImpl->Builder, Loc, DstTy, Src);
  } else if (FromFloat && ToInt) {
    if (ToTy.Signed)
      Result = createOp<arith::FPToSIOp>(PImpl->Builder, Loc, DstTy, Src);
    else
      Result = createOp<arith::FPToUIOp>(PImpl->Builder, Loc, DstTy, Src);
  } else if (FromInt && ToInt) {
    const unsigned FromBits = Src.getType().getIntOrFloatBitWidth();
    const unsigned ToBits = DstTy.getIntOrFloatBitWidth();
    if (ToBits < FromBits)
      Result = createOp<arith::TruncIOp>(PImpl->Builder, Loc, DstTy, Src);
    else if (ToBits == FromBits)
      Result = Src;
    else if (FromTy.Signed)
      Result = createOp<arith::ExtSIOp>(PImpl->Builder, Loc, DstTy, Src);
    else
      Result = createOp<arith::ExtUIOp>(PImpl->Builder, Loc, DstTy, Src);
  } else if (FromFloat && ToFloat) {
    const unsigned FromBits = Src.getType().getIntOrFloatBitWidth();
    const unsigned ToBits = DstTy.getIntOrFloatBitWidth();
    if (ToBits < FromBits)
      Result = createOp<arith::TruncFOp>(PImpl->Builder, Loc, DstTy, Src);
    else if (ToBits == FromBits)
      Result = Src;
    else
      Result = createOp<arith::ExtFOp>(PImpl->Builder, Loc, DstTy, Src);
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
  mlir::Value C = createOp<arith::ConstantOp>(PImpl->Builder, Loc, Attr);
  return PImpl->wrap(C);
}

IRValue *MLIRCodeBuilder::getConstantFP(IRType Ty, double Val) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  mlir::Type MLIRTy = toMLIRType(Ty, PImpl->Context);
  auto FTy = mlir::cast<mlir::FloatType>(MLIRTy);
  bool LosesInfo;
  llvm::APFloat APVal(Val); // double precision by default
  APVal.convert(FTy.getFloatSemantics(), llvm::APFloat::rmNearestTiesToEven,
                &LosesInfo);
  auto Attr = FloatAttr::get(FTy, APVal);
  mlir::Value C = createOp<arith::ConstantOp>(PImpl->Builder, Loc, Attr);
  return PImpl->wrap(C);
}

// ---------------------------------------------------------------------------
// loadScalar / storeScalar / allocScalar
// ---------------------------------------------------------------------------

IRValue *MLIRCodeBuilder::loadScalar(IRValue *Slot, IRType /*ValueTy*/) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  mlir::Value MemRefVal = PImpl->unwrap(Slot);
  mlir::Value Idx = createOp<arith::ConstantIndexOp>(PImpl->Builder, Loc, 0);
  mlir::Value Val =
      createOp<memref::LoadOp>(PImpl->Builder, Loc, MemRefVal, ValueRange{Idx});
  return PImpl->wrap(Val);
}

void MLIRCodeBuilder::storeScalar(IRValue *Slot, IRValue *Val) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  mlir::Value MemRefVal = PImpl->unwrap(Slot);
  mlir::Value V = PImpl->unwrap(Val);
  mlir::Value Idx = createOp<arith::ConstantIndexOp>(PImpl->Builder, Loc, 0);
  createOp<memref::StoreOp>(PImpl->Builder, Loc, V, MemRefVal, ValueRange{Idx});
}

VarAlloc MLIRCodeBuilder::allocScalar(const std::string & /*Name*/,
                                      IRType ValueTy) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  mlir::Type ElemTy = toMLIRType(ValueTy, PImpl->Context);
  // Allocate memref<1xT> (mirrors LLVM's alloca pattern).
  auto MemRefTy = MemRefType::get({1}, ElemTy);

  mlir::Value Alloca;
  {
    // Always place the alloca at the start of the entry block.
    OpBuilder::InsertionGuard Guard(PImpl->Builder);
    PImpl->Builder.setInsertionPointToStart(PImpl->EntryBlock);
    Alloca = createOp<memref::AllocaOp>(PImpl->Builder, Loc, MemRefTy);
  }

  IRType AllocTy;
  AllocTy.Kind = IRTypeKind::Pointer;
  AllocTy.ElemKind = ValueTy.Kind;

  return VarAlloc{PImpl->wrap(Alloca), ValueTy, AllocTy, 0};
}

// ---------------------------------------------------------------------------
// Structured control flow (scf.*)
// ---------------------------------------------------------------------------

void MLIRCodeBuilder::beginIf(IRValue *Cond, const char * /*File*/,
                              int /*Line*/) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  mlir::Value CondV = PImpl->unwrap(Cond);

  // Save the current insertion point for endIf to restore.
  PImpl->ScopeStack.push_back(
      {ScopeKind::IF, PImpl->Builder.saveInsertionPoint()});

  // Create scf.if with no results (mutations via memref side-effects).
  auto IfOp = createOp<scf::IfOp>(PImpl->Builder, Loc,
                                  /*resultTypes=*/TypeRange{}, CondV,
                                  /*withElseRegion=*/false);

  // Set insertion point to the start of the then region.
  PImpl->Builder.setInsertionPointToStart(&IfOp.getThenRegion().front());
}

void MLIRCodeBuilder::endIf() {
  // Ensure the then region has a scf.yield terminator.
  Block *CurBlock = PImpl->Builder.getInsertionBlock();
  if (CurBlock->empty() || !CurBlock->back().hasTrait<OpTrait::IsTerminator>())
    createOp<scf::YieldOp>(PImpl->Builder, PImpl->Builder.getUnknownLoc());

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
  mlir::Value LB = createOp<arith::IndexCastOp>(PImpl->Builder, Loc,
                                                PImpl->Builder.getIndexType(),
                                                PImpl->unwrap(InitVal));
  mlir::Value UB = createOp<arith::IndexCastOp>(PImpl->Builder, Loc,
                                                PImpl->Builder.getIndexType(),
                                                PImpl->unwrap(UpperBoundVal));
  mlir::Value Step = createOp<arith::IndexCastOp>(PImpl->Builder, Loc,
                                                  PImpl->Builder.getIndexType(),
                                                  PImpl->unwrap(IncVal));

  // Save insertion point for endFor.
  PImpl->ScopeStack.push_back(
      {ScopeKind::FOR, PImpl->Builder.saveInsertionPoint()});

  // Create scf.for with no iter_args (mutation via memref side-effects).
  auto ForOp = createOp<scf::ForOp>(PImpl->Builder, Loc, LB, UB, Step);

  // Set insertion point inside the body.
  PImpl->Builder.setInsertionPointToStart(ForOp.getBody());

  // Cast the index induction variable back to the original integer type
  // and store into IterSlot so user code can read it.
  mlir::Value IV = ForOp.getInductionVar();
  mlir::Type IterMLIRTy = toMLIRType(IterTy, PImpl->Context);
  mlir::Value TypedIV =
      createOp<arith::IndexCastOp>(PImpl->Builder, Loc, IterMLIRTy, IV);
  mlir::Value Idx = createOp<arith::ConstantIndexOp>(PImpl->Builder, Loc, 0);
  mlir::Value SlotV = PImpl->unwrap(IterSlot);
  createOp<memref::StoreOp>(PImpl->Builder, Loc, TypedIV, SlotV,
                            ValueRange{Idx});
}

void MLIRCodeBuilder::endFor() {
  // Ensure the scf.for body has a scf.yield terminator.
  Block *CurBlock = PImpl->Builder.getInsertionBlock();
  if (CurBlock->empty() || !CurBlock->back().hasTrait<OpTrait::IsTerminator>())
    createOp<scf::YieldOp>(PImpl->Builder, PImpl->Builder.getUnknownLoc());

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
  auto WhileOp = createOp<scf::WhileOp>(PImpl->Builder, Loc,
                                        /*resultTypes=*/TypeRange{},
                                        /*operands=*/ValueRange{});

  // --- Fill the "before" region (condition). ---
  Block *BeforeBlock = PImpl->Builder.createBlock(&WhileOp.getBefore());
  PImpl->Builder.setInsertionPointToEnd(BeforeBlock);

  // Call CondFn to emit the condition IR into the before region.
  IRValue *CondIRV = CondFn();
  mlir::Value CondV = PImpl->unwrap(CondIRV);

  // Terminate the before region with scf.condition.
  createOp<scf::ConditionOp>(PImpl->Builder, Loc, CondV,
                             /*args=*/ValueRange{});

  // --- Prepare the "after" region (body). ---
  Block *AfterBlock = PImpl->Builder.createBlock(&WhileOp.getAfter());
  PImpl->Builder.setInsertionPointToStart(AfterBlock);
  // User body code will emit here between beginWhile/endWhile.
}

void MLIRCodeBuilder::endWhile() {
  // Ensure the after region has a scf.yield terminator.
  Block *CurBlock = PImpl->Builder.getInsertionBlock();
  if (CurBlock->empty() || !CurBlock->back().hasTrait<OpTrait::IsTerminator>())
    createOp<scf::YieldOp>(PImpl->Builder, PImpl->Builder.getUnknownLoc());

  auto Scope = PImpl->ScopeStack.pop_back_val();
  PImpl->Builder.restoreInsertionPoint(Scope.SavedIP);
}

IRValue *MLIRCodeBuilder::createCmp(CmpOp Op, IRValue *LHS, IRValue *RHS,
                                    IRType Ty) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  mlir::Value L = PImpl->unwrap(LHS);
  mlir::Value R = PImpl->unwrap(RHS);
  mlir::Value Result;

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
    Result = createOp<arith::CmpFOp>(PImpl->Builder, Loc, Pred, L, R);
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
    Result = createOp<arith::CmpIOp>(PImpl->Builder, Loc, Pred, L, R);
  }
  return PImpl->wrap(Result);
}

IRValue *MLIRCodeBuilder::createAnd(IRValue *LHS, IRValue *RHS) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  mlir::Value Result = createOp<arith::AndIOp>(
      PImpl->Builder, Loc, PImpl->unwrap(LHS), PImpl->unwrap(RHS));
  return PImpl->wrap(Result);
}

IRValue *MLIRCodeBuilder::createOr(IRValue *LHS, IRValue *RHS) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  mlir::Value Result = createOp<arith::OrIOp>(
      PImpl->Builder, Loc, PImpl->unwrap(LHS), PImpl->unwrap(RHS));
  return PImpl->wrap(Result);
}

IRValue *MLIRCodeBuilder::createXor(IRValue *LHS, IRValue *RHS) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  mlir::Value Result = createOp<arith::XOrIOp>(
      PImpl->Builder, Loc, PImpl->unwrap(LHS), PImpl->unwrap(RHS));
  return PImpl->wrap(Result);
}

IRValue *MLIRCodeBuilder::createNot(IRValue *Val) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  // Logical NOT for i1: xor with true.
  auto TrueAttr =
      IntegerAttr::get(mlir::IntegerType::get(&PImpl->Context, 1), 1);
  mlir::Value TrueVal =
      createOp<arith::ConstantOp>(PImpl->Builder, Loc, TrueAttr);
  mlir::Value Result =
      createOp<arith::XOrIOp>(PImpl->Builder, Loc, PImpl->unwrap(Val), TrueVal);
  return PImpl->wrap(Result);
}

IRValue *MLIRCodeBuilder::createLoad(IRType Ty, IRValue *Ptr,
                                     const std::string & /*Name*/) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  mlir::Value PtrV = PImpl->unwrap(Ptr);
  mlir::Type ExpectedTy = toMLIRScalarType(Ty.Kind, PImpl->Context);
  auto LLVMPointerTy = mlir::LLVM::LLVMPointerType::get(&PImpl->Context);

  // Raw-address case: Ptr is a first-class pointer value; load directly.
  if (mlir::isa<mlir::LLVM::LLVMPointerType>(PtrV.getType())) {
    mlir::Value Val =
        createOp<mlir::LLVM::LoadOp>(PImpl->Builder, Loc, ExpectedTy, PtrV);
    return PImpl->wrap(Val);
  }

  // Scalar-slot case: Ptr is a mutable scalar slot represented as
  // memref<1xT>; load slot[0].
  if (Impl::isScalarSlotType(PtrV.getType())) {
    auto SlotTy = cast<MemRefType>(PtrV.getType());
    mlir::Value Zero = createOp<arith::ConstantIndexOp>(PImpl->Builder, Loc, 0);

    // Direct scalar slot: memref<1xElemTy> where ElemTy is the scalar type.
    if (SlotTy.getElementType() == ExpectedTy) {
      mlir::Value Val =
          createOp<memref::LoadOp>(PImpl->Builder, Loc, PtrV, ValueRange{Zero});
      return PImpl->wrap(Val);
    }

    // Address slot: memref<1x!llvm.ptr> holding an address of an element.
    if (SlotTy.getElementType() == LLVMPointerTy) {
      mlir::Value Addr =
          createOp<memref::LoadOp>(PImpl->Builder, Loc, PtrV, ValueRange{Zero});
      mlir::Value Val =
          createOp<mlir::LLVM::LoadOp>(PImpl->Builder, Loc, ExpectedTy, Addr);
      return PImpl->wrap(Val);
    }

    reportFatalError(
        "MLIRCodeBuilder::createLoad: unsupported scalar slot type");
  }

  reportFatalError("MLIRCodeBuilder::createLoad: unsupported Ptr form");
}

void MLIRCodeBuilder::createStore(IRValue *Val, IRValue *Ptr) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  mlir::Value V = PImpl->unwrap(Val);
  mlir::Value PtrV = PImpl->unwrap(Ptr);

  // Raw-address case: Ptr is a first-class pointer value; store directly.
  if (mlir::isa<mlir::LLVM::LLVMPointerType>(PtrV.getType())) {
    createOp<mlir::LLVM::StoreOp>(PImpl->Builder, Loc, V, PtrV);
    return;
  }

  // Scalar-slot case: Ptr is memref<1xT>; store into slot[0].
  if (Impl::isScalarSlotType(PtrV.getType())) {
    auto SlotTy = cast<MemRefType>(PtrV.getType());
    mlir::Value Zero = createOp<arith::ConstantIndexOp>(PImpl->Builder, Loc, 0);
    auto LLVMPointerTy = mlir::LLVM::LLVMPointerType::get(&PImpl->Context);

    // Direct scalar slot: memref<1xElemTy> where ElemTy matches V's type.
    if (SlotTy.getElementType() == V.getType()) {
      createOp<memref::StoreOp>(PImpl->Builder, Loc, V, PtrV, ValueRange{Zero});
      return;
    }

    // Address slot: memref<1x!llvm.ptr> holding an address of an element.
    if (SlotTy.getElementType() == LLVMPointerTy) {
      mlir::Value Addr =
          createOp<memref::LoadOp>(PImpl->Builder, Loc, PtrV, ValueRange{Zero});
      createOp<mlir::LLVM::StoreOp>(PImpl->Builder, Loc, V, Addr);
      return;
    }

    reportFatalError(
        "MLIRCodeBuilder::createStore: unsupported scalar slot type");
  }

  reportFatalError("MLIRCodeBuilder::createStore: unsupported Ptr form");
}

IRValue *MLIRCodeBuilder::createBitCast(IRValue *V, IRType DestTy) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  mlir::Value Src = PImpl->unwrap(V);
  mlir::Type SrcTy = Src.getType();
  mlir::Type DstTy = PImpl->toScalarMLIRType(DestTy);

  if (!Impl::isScalarIntOrFloat(SrcTy) || !Impl::isScalarIntOrFloat(DstTy))
    reportFatalError("createBitCast: only scalar int/float supported");

  if (SrcTy == DstTy)
    return V;

  const unsigned BWSrc = Impl::getBitWidthOrZero(SrcTy);
  const unsigned BWDst = Impl::getBitWidthOrZero(DstTy);
  if (BWSrc == 0 || BWDst == 0 || BWSrc != BWDst)
    reportFatalError("createBitCast: source and destination must have equal "
                     "non-zero bitwidth");

  mlir::Value Res = createOp<arith::BitcastOp>(PImpl->Builder, Loc, DstTy, Src);
  return PImpl->wrap(Res);
}

IRValue *MLIRCodeBuilder::createZExt(IRValue *V, IRType DestTy) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  mlir::Value Src = PImpl->unwrap(V);
  mlir::Type SrcTy = Src.getType();
  mlir::Type DstTy = PImpl->toScalarMLIRType(DestTy);

  if (mlir::isa<mlir::FloatType>(SrcTy) || mlir::isa<mlir::FloatType>(DstTy))
    reportFatalError("createZExt: float types are not supported");

  const bool SrcIsInt = mlir::isa<mlir::IntegerType>(SrcTy);
  const bool DstIsInt = mlir::isa<mlir::IntegerType>(DstTy);
  const bool SrcIsIndex = mlir::isa<mlir::IndexType>(SrcTy);
  const bool DstIsIndex = mlir::isa<mlir::IndexType>(DstTy);
  if ((!SrcIsInt && !SrcIsIndex) || (!DstIsInt && !DstIsIndex))
    reportFatalError("createZExt: only integer/index types are supported");

  if (SrcTy == DstTy)
    return V;

  if (SrcIsInt && DstIsInt) {
    auto SrcIntTy = mlir::cast<mlir::IntegerType>(SrcTy);
    auto DstIntTy = mlir::cast<mlir::IntegerType>(DstTy);
    if (DstIntTy.getWidth() <= SrcIntTy.getWidth())
      reportFatalError("createZExt: destination integer must be wider than "
                       "source integer");
    mlir::Value Res = createOp<arith::ExtUIOp>(PImpl->Builder, Loc, DstTy, Src);
    return PImpl->wrap(Res);
  }

  if ((SrcIsIndex && DstIsInt) || (SrcIsInt && DstIsIndex)) {
    mlir::Value Res =
        createOp<arith::IndexCastUIOp>(PImpl->Builder, Loc, DstTy, Src);
    return PImpl->wrap(Res);
  }

  reportFatalError("createZExt: unsupported type combination");
}

VarAlloc MLIRCodeBuilder::getElementPtr(IRValue *Base, IRType /*BaseTy*/,
                                        IRValue *Index, IRType ElemTy) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  auto I64Ty = mlir::IntegerType::get(&PImpl->Context, 64);

  // Normalize element index to i64 for LLVM's `getelementptr`.
  mlir::Value RawIdx = PImpl->unwrap(Index);
  mlir::Value IdxI64;
  if (mlir::isa<mlir::IndexType>(RawIdx.getType())) {
    IdxI64 = createOp<arith::IndexCastUIOp>(PImpl->Builder, Loc, I64Ty, RawIdx);
  } else if (auto IntTy = dyn_cast<mlir::IntegerType>(RawIdx.getType())) {
    if (IntTy.getWidth() == 64) {
      IdxI64 = RawIdx;
    } else if (IntTy.getWidth() < 64) {
      IdxI64 = createOp<arith::ExtUIOp>(PImpl->Builder, Loc, I64Ty, RawIdx);
    } else {
      IdxI64 = createOp<arith::TruncIOp>(PImpl->Builder, Loc, I64Ty, RawIdx);
    }
  } else {
    reportFatalError("getElementPtr: expected integer-like index");
  }

  return PImpl->emitElementPtrSlot(Base, IdxI64, ElemTy);
}

VarAlloc MLIRCodeBuilder::getElementPtr(IRValue *Base, IRType /*BaseTy*/,
                                        size_t Index, IRType ElemTy) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  mlir::Value IdxI64 =
      createOp<arith::ConstantIntOp>(PImpl->Builder, Loc, Index, 64);
  return PImpl->emitElementPtrSlot(Base, IdxI64, ElemTy);
}

IRValue *MLIRCodeBuilder::createCall(const std::string &FName, IRType RetTy,
                                     const std::vector<IRType> &ArgTys,
                                     const std::vector<IRValue *> &Args) {
  if (ArgTys.size() != Args.size())
    reportFatalError("createCall: ArgTys.size() must match Args.size()");

  if (PImpl->isRawPointerAbiType(RetTy))
    reportFatalError("createCall: pointer return types are unsupported");

  if (!PImpl->CurrentFuncOp)
    reportFatalError("createCall: no active function context");

  const bool InDeviceContext = !isHostTargetModel(TargetModel);

  func::FuncOp DeviceInternalCallee;
  bool CalleeIsDeviceInternal = false;
  if (InDeviceContext) {
    DeviceInternalCallee = PImpl->lookupDeviceInternalFunc(FName);
    if (DeviceInternalCallee) {
      CalleeIsDeviceInternal = true;
    } else {
      reportFatalError("createCall: unresolved device callee " + FName +
                       " (define func.func helper before calling it)");
    }
  }

  llvm::SmallVector<mlir::Type> MLIRArgTys;
  MLIRArgTys.reserve(ArgTys.size());
  for (const auto &ArgTy : ArgTys)
    MLIRArgTys.push_back(toMLIRType(ArgTy, PImpl->Context));

  llvm::SmallVector<mlir::Type> ResultTys;
  if (RetTy.Kind != IRTypeKind::Void)
    ResultTys.push_back(toMLIRType(RetTy, PImpl->Context));

  auto FTy = PImpl->Builder.getFunctionType(MLIRArgTys, ResultTys);

  llvm::SmallVector<mlir::Value> CallArgs;
  CallArgs.reserve(Args.size());
  for (IRValue *Arg : Args)
    CallArgs.push_back(PImpl->unwrap(Arg));

  auto Loc = PImpl->Builder.getUnknownLoc();
  func::CallOp Call;
  if (InDeviceContext) {
    if (!CalleeIsDeviceInternal)
      reportFatalError("createCall: expected internal device callee");
    if (DeviceInternalCallee.getFunctionType() != FTy)
      reportFatalError("createCall: device callee type mismatch for " + FName);
    Call = createOp<func::CallOp>(PImpl->Builder, Loc, FName, FTy.getResults(),
                                  ValueRange{CallArgs});
  } else {
    auto Callee = PImpl->getOrCreateFunc(FName, FTy);
    (void)Callee;
    Call = createOp<func::CallOp>(PImpl->Builder, Loc, FName, FTy.getResults(),
                                  ValueRange{CallArgs});
  }

  if (RetTy.Kind == IRTypeKind::Void)
    return nullptr;
  return PImpl->wrap(Call.getResult(0));
}

IRValue *MLIRCodeBuilder::createCall(const std::string &FName, IRType RetTy) {
  return createCall(FName, RetTy, {}, {});
}

IRValue *MLIRCodeBuilder::emitIntrinsic(const std::string &Name, IRType RetTy,
                                        const std::vector<IRValue *> &Args) {
  auto Loc = PImpl->Builder.getUnknownLoc();

  auto RequireArgCount = [&](size_t Expected) {
    if (Args.size() != Expected)
      reportFatalError("MLIRCodeBuilder::emitIntrinsic: wrong argument count "
                       "for " +
                       Name);
  };

  auto RequireFloatRetTy = [&]() {
    if (!isFloatingPointKind(RetTy))
      reportFatalError("MLIRCodeBuilder::emitIntrinsic: expected float return "
                       "type for " +
                       Name);
  };

  auto UnwrapArg = [&](size_t I) -> mlir::Value {
    mlir::Value V = PImpl->unwrap(Args[I]);
    if (V.getType() != toMLIRType(RetTy, PImpl->Context))
      reportFatalError("MLIRCodeBuilder::emitIntrinsic: argument type mismatch "
                       "for " +
                       Name);
    return V;
  };

  if (Name == "sinf") {
    RequireArgCount(1);
    RequireFloatRetTy();
    return PImpl->wrap(
        createOp<math::SinOp>(PImpl->Builder, Loc, UnwrapArg(0)));
  }
  if (Name == "cosf") {
    RequireArgCount(1);
    RequireFloatRetTy();
    return PImpl->wrap(
        createOp<math::CosOp>(PImpl->Builder, Loc, UnwrapArg(0)));
  }
  if (Name == "expf") {
    RequireArgCount(1);
    RequireFloatRetTy();
    return PImpl->wrap(
        createOp<math::ExpOp>(PImpl->Builder, Loc, UnwrapArg(0)));
  }
  if (Name == "logf") {
    RequireArgCount(1);
    RequireFloatRetTy();
    return PImpl->wrap(
        createOp<math::LogOp>(PImpl->Builder, Loc, UnwrapArg(0)));
  }
  if (Name == "sqrtf") {
    RequireArgCount(1);
    RequireFloatRetTy();
    return PImpl->wrap(
        createOp<math::SqrtOp>(PImpl->Builder, Loc, UnwrapArg(0)));
  }
  if (Name == "truncf") {
    RequireArgCount(1);
    RequireFloatRetTy();
    return PImpl->wrap(
        createOp<math::TruncOp>(PImpl->Builder, Loc, UnwrapArg(0)));
  }
  if (Name == "powf") {
    RequireArgCount(2);
    RequireFloatRetTy();
    return PImpl->wrap(createOp<math::PowFOp>(PImpl->Builder, Loc, UnwrapArg(0),
                                              UnwrapArg(1)));
  }
  if (Name == "fabsf" || Name == "absf") {
    RequireArgCount(1);
    RequireFloatRetTy();
    auto Arg = UnwrapArg(0);
    auto Neg = createOp<arith::NegFOp>(PImpl->Builder, Loc, Arg);
    return PImpl->wrap(
        createOp<arith::MaximumFOp>(PImpl->Builder, Loc, Arg, Neg));
  }

  reportFatalError("MLIRCodeBuilder::emitIntrinsic: unsupported intrinsic " +
                   Name);
}

IRValue *MLIRCodeBuilder::emitBuiltin(const std::string &Name, IRType RetTy,
                                      const std::vector<IRValue *> &Args) {
  if (!Args.empty())
    reportFatalError("MLIRCodeBuilder::emitBuiltin: builtins do not accept "
                     "arguments");

  if (isHostTargetModel(TargetModel))
    reportFatalError("MLIRCodeBuilder::emitBuiltin: builtin only valid in GPU "
                     "device code");
  if (!PImpl->CurrentFuncOp)
    reportFatalError(
        "MLIRCodeBuilder::emitBuiltin: there is no active function "
        "context");

  auto Loc = PImpl->Builder.getUnknownLoc();
  auto EmitHipImplicitArgPtr = [&]() -> mlir::Value {
    constexpr unsigned ConstantAddressSpace = 4;
    auto PtrTy =
        mlir::LLVM::LLVMPointerType::get(&PImpl->Context, ConstantAddressSpace);
#if LLVM_VERSION_MAJOR >= 20
    auto Call = createOp<mlir::LLVM::CallIntrinsicOp>(
        PImpl->Builder, Loc, PtrTy,
        PImpl->Builder.getStringAttr("llvm.amdgcn.implicitarg.ptr"),
        ValueRange{});
#else
    auto Call = createOp<mlir::LLVM::CallIntrinsicOp>(
        PImpl->Builder, Loc, TypeRange{PtrTy}, "llvm.amdgcn.implicitarg.ptr",
        ValueRange{});
#endif
    return Call.getResult(0);
  };
  auto EmitLLVMConstI64 = [&](int64_t V) -> mlir::Value {
    auto I64Ty = mlir::IntegerType::get(&PImpl->Context, 64);
    return createOp<mlir::LLVM::ConstantOp>(
        PImpl->Builder, Loc, I64Ty, PImpl->Builder.getI64IntegerAttr(V));
  };
  auto EmitImplicitArgLoad = [&](mlir::Type ElemTy,
                                 int64_t Offset) -> mlir::Value {
    constexpr unsigned ConstantAddressSpace = 4;
    auto PtrTy =
        mlir::LLVM::LLVMPointerType::get(&PImpl->Context, ConstantAddressSpace);
    mlir::Value ImplicitArgPtr = EmitHipImplicitArgPtr();
    mlir::Value OffsetVal = EmitLLVMConstI64(Offset);
    auto GEP =
        createOp<mlir::LLVM::GEPOp>(PImpl->Builder, Loc, PtrTy, ElemTy,
                                    ImplicitArgPtr, ValueRange{OffsetVal});
    return createOp<mlir::LLVM::LoadOp>(PImpl->Builder, Loc, ElemTy, GEP);
  };
  auto CastI32ToRetTy = [&](mlir::Value I32V) -> IRValue * {
    if (RetTy.Kind == IRTypeKind::Int32)
      return PImpl->wrap(I32V);
    if (RetTy.Kind == IRTypeKind::Int64) {
      auto I64Ty = mlir::IntegerType::get(&PImpl->Context, 64);
      auto Cast = createOp<arith::ExtUIOp>(PImpl->Builder, Loc, I64Ty, I32V);
      return PImpl->wrap(Cast);
    }
    reportFatalError("MLIRCodeBuilder::emitBuiltin: unsupported return type "
                     "for " +
                     Name);
  };
  auto CastIndexToRetTy = [&](mlir::Value IndexV) -> IRValue * {
    if (RetTy.Kind == IRTypeKind::Int32 || RetTy.Kind == IRTypeKind::Int64) {
      // gpu dialect IDs are target-agnostic index values; cast at the builder
      // boundary to preserve the frontend's integer builtin API.
      auto DstTy = toMLIRType(RetTy, PImpl->Context);
      auto Cast =
          createOp<arith::IndexCastUIOp>(PImpl->Builder, Loc, DstTy, IndexV);
      return PImpl->wrap(Cast);
    }
    reportFatalError("MLIRCodeBuilder::emitBuiltin: unsupported return type "
                     "for " +
                     Name);
  };

  // gpu dialect is architecture-agnostic; target-specific lowering to
  // NVVM/ROCDL intrinsics happens in later conversion passes.
  if (Name == "threadIdx.x")
    return CastIndexToRetTy(
        createOp<gpu::ThreadIdOp>(PImpl->Builder, Loc, gpu::Dimension::x));
  if (Name == "threadIdx.y")
    return CastIndexToRetTy(
        createOp<gpu::ThreadIdOp>(PImpl->Builder, Loc, gpu::Dimension::y));
  if (Name == "threadIdx.z")
    return CastIndexToRetTy(
        createOp<gpu::ThreadIdOp>(PImpl->Builder, Loc, gpu::Dimension::z));

  if (Name == "blockIdx.x")
    return CastIndexToRetTy(
        createOp<gpu::BlockIdOp>(PImpl->Builder, Loc, gpu::Dimension::x));
  if (Name == "blockIdx.y")
    return CastIndexToRetTy(
        createOp<gpu::BlockIdOp>(PImpl->Builder, Loc, gpu::Dimension::y));
  if (Name == "blockIdx.z")
    return CastIndexToRetTy(
        createOp<gpu::BlockIdOp>(PImpl->Builder, Loc, gpu::Dimension::z));

  // For HIP, avoid lowering block/grid dims via OCKL runtime calls
  // (__ockl_get_local_size/__ockl_get_num_groups). Emit implicitarg loads
  // directly, mirroring the LLVM backend implementation.
  if (TargetModel == TargetModelType::HIP) {
    auto I16Ty = mlir::IntegerType::get(&PImpl->Context, 16);
    auto I32Ty = mlir::IntegerType::get(&PImpl->Context, 32);

    auto EmitBlockDim = [&](int64_t Offset) -> IRValue * {
      // HIP block dimensions are encoded as i16 in the implicit argument area;
      // zero-extend to i32 to match the frontend builtin contract.
      mlir::Value V16 = EmitImplicitArgLoad(I16Ty, Offset);
      mlir::Value V32 =
          createOp<arith::ExtUIOp>(PImpl->Builder, Loc, I32Ty, V16);
      return CastI32ToRetTy(V32);
    };
    auto EmitGridDim = [&](int64_t Offset) -> IRValue * {
      // HIP grid dimensions are encoded as i32 in the implicit argument area.
      mlir::Value V32 = EmitImplicitArgLoad(I32Ty, Offset);
      return CastI32ToRetTy(V32);
    };

    if (Name == "blockDim.x")
      return EmitBlockDim(/*Offset=*/6);
    if (Name == "blockDim.y")
      return EmitBlockDim(/*Offset=*/7);
    if (Name == "blockDim.z")
      return EmitBlockDim(/*Offset=*/8);

    if (Name == "gridDim.x")
      return EmitGridDim(/*Offset=*/0);
    if (Name == "gridDim.y")
      return EmitGridDim(/*Offset=*/1);
    if (Name == "gridDim.z")
      return EmitGridDim(/*Offset=*/2);
  }

  if (Name == "blockDim.x")
    return CastIndexToRetTy(
        createOp<gpu::BlockDimOp>(PImpl->Builder, Loc, gpu::Dimension::x));
  if (Name == "blockDim.y")
    return CastIndexToRetTy(
        createOp<gpu::BlockDimOp>(PImpl->Builder, Loc, gpu::Dimension::y));
  if (Name == "blockDim.z")
    return CastIndexToRetTy(
        createOp<gpu::BlockDimOp>(PImpl->Builder, Loc, gpu::Dimension::z));

  if (Name == "gridDim.x")
    return CastIndexToRetTy(
        createOp<gpu::GridDimOp>(PImpl->Builder, Loc, gpu::Dimension::x));
  if (Name == "gridDim.y")
    return CastIndexToRetTy(
        createOp<gpu::GridDimOp>(PImpl->Builder, Loc, gpu::Dimension::y));
  if (Name == "gridDim.z")
    return CastIndexToRetTy(
        createOp<gpu::GridDimOp>(PImpl->Builder, Loc, gpu::Dimension::z));

  if (Name == "syncThreads") {
    createOp<gpu::BarrierOp>(PImpl->Builder, Loc);
    return nullptr;
  }

  reportFatalError("MLIRCodeBuilder::emitBuiltin: unsupported builtin " + Name);
}

IRValue *MLIRCodeBuilder::loadAddress(IRValue *Slot, IRType /*AllocTy*/) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  mlir::Value SlotV = PImpl->unwrap(Slot);
  auto SlotTy = dyn_cast<MemRefType>(SlotV.getType());
  if (!SlotTy || SlotTy.getRank() != 1 || SlotTy.getShape()[0] != 1 ||
      !mlir::isa<mlir::LLVM::LLVMPointerType>(SlotTy.getElementType()))
    reportFatalError(
        "loadAddress: expected pointer slot (memref<1x!llvm.ptr<...>>)");

  mlir::Value Zero = createOp<arith::ConstantIndexOp>(PImpl->Builder, Loc, 0);
  return PImpl->wrap(
      createOp<memref::LoadOp>(PImpl->Builder, Loc, SlotV, ValueRange{Zero}));
}

void MLIRCodeBuilder::storeAddress(IRValue *Slot, IRValue *Addr) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  auto I64Ty = mlir::IntegerType::get(&PImpl->Context, 64);

  mlir::Value LhsSlot = PImpl->unwrap(Slot);
  auto LhsSlotTy = dyn_cast<MemRefType>(LhsSlot.getType());
  if (!LhsSlotTy || LhsSlotTy.getRank() != 1 || LhsSlotTy.getShape()[0] != 1 ||
      !mlir::isa<mlir::LLVM::LLVMPointerType>(LhsSlotTy.getElementType()))
    reportFatalError(
        "storeAddress: expected pointer slot (memref<1x!llvm.ptr<...>>)");
  auto SlotPtrTy =
      mlir::cast<mlir::LLVM::LLVMPointerType>(LhsSlotTy.getElementType());

  mlir::Value AddrV = PImpl->unwrap(Addr);
  mlir::Value PtrVal;
  if (auto AddrPtrTy = dyn_cast<mlir::LLVM::LLVMPointerType>(AddrV.getType())) {
    PtrVal = AddrV;
    if (AddrPtrTy.getAddressSpace() != SlotPtrTy.getAddressSpace())
      PtrVal = createOp<mlir::LLVM::AddrSpaceCastOp>(PImpl->Builder, Loc,
                                                     SlotPtrTy, PtrVal);
  } else if (mlir::isa<MemRefType>(AddrV.getType())) {
    // Form a raw pointer value to element 0 of the memref.
    unsigned AddrSpace = 0;
    if (auto AddrMemRefTy = dyn_cast<MemRefType>(AddrV.getType())) {
      if (auto MemSpaceAttr = dyn_cast_or_null<mlir::IntegerAttr>(
              AddrMemRefTy.getMemorySpace()))
        AddrSpace = static_cast<unsigned>(MemSpaceAttr.getInt());
    }
    auto PtrTy = mlir::LLVM::LLVMPointerType::get(&PImpl->Context, AddrSpace);
    mlir::Value BaseAddrAsIndex =
        createOp<memref::ExtractAlignedPointerAsIndexOp>(PImpl->Builder, Loc,
                                                         AddrV);
    mlir::Value BaseAddrI64 = createOp<arith::IndexCastUIOp>(
        PImpl->Builder, Loc, I64Ty, BaseAddrAsIndex);
    PtrVal = createOp<mlir::LLVM::IntToPtrOp>(PImpl->Builder, Loc, PtrTy,
                                              BaseAddrI64);
    if (PtrTy.getAddressSpace() != SlotPtrTy.getAddressSpace())
      PtrVal = createOp<mlir::LLVM::AddrSpaceCastOp>(PImpl->Builder, Loc,
                                                     SlotPtrTy, PtrVal);
  } else {
    reportFatalError("storeAddress: unsupported address value kind");
  }

  mlir::Value Zero = createOp<arith::ConstantIndexOp>(PImpl->Builder, Loc, 0);
  createOp<memref::StoreOp>(PImpl->Builder, Loc, PtrVal, LhsSlot,
                            ValueRange{Zero});
}

IRValue *MLIRCodeBuilder::createAtomicAdd(IRValue *Addr, IRValue *Val) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  mlir::Value AddrV = PImpl->unwrap(Addr);
  mlir::Value Ptr;
  if (mlir::isa<mlir::LLVM::LLVMPointerType>(AddrV.getType())) {
    Ptr = AddrV;
  } else if (auto SlotTy = dyn_cast<MemRefType>(AddrV.getType());
             SlotTy && SlotTy.getRank() == 1 && SlotTy.getShape()[0] == 1 &&
             mlir::isa<mlir::LLVM::LLVMPointerType>(SlotTy.getElementType())) {
    mlir::Value Zero = createOp<arith::ConstantIndexOp>(PImpl->Builder, Loc, 0);
    Ptr =
        createOp<memref::LoadOp>(PImpl->Builder, Loc, AddrV, ValueRange{Zero});
  } else {
    reportFatalError("createAtomicAdd: expected !llvm.ptr or "
                     "memref<1x!llvm.ptr> address slot");
  }

  mlir::Value V = PImpl->unwrap(Val);
  if (mlir::isa<mlir::FloatType>(V.getType()) &&
      !isSupportedAtomicFloatType(V.getType()))
    reportFatalError("createAtomicAdd: unsupported floating-point atomic type");

  mlir::LLVM::AtomicBinOp BinOp = mlir::isa<mlir::FloatType>(V.getType())
                                      ? mlir::LLVM::AtomicBinOp::fadd
                                      : mlir::LLVM::AtomicBinOp::add;
  auto Atomic = createOp<mlir::LLVM::AtomicRMWOp>(
      PImpl->Builder, Loc, V.getType(), BinOp, Ptr, V,
      mlir::LLVM::AtomicOrdering::seq_cst,
      /*syncscope=*/mlir::StringAttr{}, /*alignment=*/mlir::IntegerAttr{},
      /*volatile_=*/false, /*access_groups=*/mlir::ArrayAttr{},
      /*alias_scopes=*/mlir::ArrayAttr{}, /*noalias_scopes=*/mlir::ArrayAttr{},
      /*tbaa=*/mlir::ArrayAttr{});
  return PImpl->wrap(Atomic.getResult());
}

IRValue *MLIRCodeBuilder::createAtomicSub(IRValue *Addr, IRValue *Val) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  mlir::Value AddrV = PImpl->unwrap(Addr);
  mlir::Value Ptr;
  if (mlir::isa<mlir::LLVM::LLVMPointerType>(AddrV.getType())) {
    Ptr = AddrV;
  } else if (auto SlotTy = dyn_cast<MemRefType>(AddrV.getType());
             SlotTy && SlotTy.getRank() == 1 && SlotTy.getShape()[0] == 1 &&
             mlir::isa<mlir::LLVM::LLVMPointerType>(SlotTy.getElementType())) {
    mlir::Value Zero = createOp<arith::ConstantIndexOp>(PImpl->Builder, Loc, 0);
    Ptr =
        createOp<memref::LoadOp>(PImpl->Builder, Loc, AddrV, ValueRange{Zero});
  } else {
    reportFatalError("createAtomicSub: expected !llvm.ptr or "
                     "memref<1x!llvm.ptr> address slot");
  }

  mlir::Value V = PImpl->unwrap(Val);
  if (mlir::isa<mlir::FloatType>(V.getType()) &&
      !isSupportedAtomicFloatType(V.getType()))
    reportFatalError("createAtomicSub: unsupported floating-point atomic type");

  mlir::LLVM::AtomicBinOp BinOp = mlir::isa<mlir::FloatType>(V.getType())
                                      ? mlir::LLVM::AtomicBinOp::fsub
                                      : mlir::LLVM::AtomicBinOp::sub;
  auto Atomic = createOp<mlir::LLVM::AtomicRMWOp>(
      PImpl->Builder, Loc, V.getType(), BinOp, Ptr, V,
      mlir::LLVM::AtomicOrdering::seq_cst,
      /*syncscope=*/mlir::StringAttr{}, /*alignment=*/mlir::IntegerAttr{},
      /*volatile_=*/false, /*access_groups=*/mlir::ArrayAttr{},
      /*alias_scopes=*/mlir::ArrayAttr{}, /*noalias_scopes=*/mlir::ArrayAttr{},
      /*tbaa=*/mlir::ArrayAttr{});
  return PImpl->wrap(Atomic.getResult());
}

IRValue *MLIRCodeBuilder::createAtomicMax(IRValue *Addr, IRValue *Val) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  mlir::Value AddrV = PImpl->unwrap(Addr);
  mlir::Value Ptr;
  if (mlir::isa<mlir::LLVM::LLVMPointerType>(AddrV.getType())) {
    Ptr = AddrV;
  } else if (auto SlotTy = dyn_cast<MemRefType>(AddrV.getType());
             SlotTy && SlotTy.getRank() == 1 && SlotTy.getShape()[0] == 1 &&
             mlir::isa<mlir::LLVM::LLVMPointerType>(SlotTy.getElementType())) {
    mlir::Value Zero = createOp<arith::ConstantIndexOp>(PImpl->Builder, Loc, 0);
    Ptr =
        createOp<memref::LoadOp>(PImpl->Builder, Loc, AddrV, ValueRange{Zero});
  } else {
    reportFatalError("createAtomicMax: expected !llvm.ptr or "
                     "memref<1x!llvm.ptr> address slot");
  }

  mlir::Value V = PImpl->unwrap(Val);
  if (mlir::isa<mlir::FloatType>(V.getType()) &&
      !isSupportedAtomicFloatType(V.getType()))
    reportFatalError("createAtomicMax: unsupported floating-point atomic type");

  mlir::LLVM::AtomicBinOp BinOp;
  if (mlir::isa<mlir::FloatType>(V.getType()))
    BinOp = mlir::LLVM::AtomicBinOp::fmax;
  else
    BinOp = mlir::LLVM::AtomicBinOp::max;

  auto Atomic = createOp<mlir::LLVM::AtomicRMWOp>(
      PImpl->Builder, Loc, V.getType(), BinOp, Ptr, V,
      mlir::LLVM::AtomicOrdering::seq_cst,
      /*syncscope=*/mlir::StringAttr{}, /*alignment=*/mlir::IntegerAttr{},
      /*volatile_=*/false, /*access_groups=*/mlir::ArrayAttr{},
      /*alias_scopes=*/mlir::ArrayAttr{}, /*noalias_scopes=*/mlir::ArrayAttr{},
      /*tbaa=*/mlir::ArrayAttr{});
  return PImpl->wrap(Atomic.getResult());
}

IRValue *MLIRCodeBuilder::createAtomicMin(IRValue *Addr, IRValue *Val) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  mlir::Value AddrV = PImpl->unwrap(Addr);
  mlir::Value Ptr;
  if (mlir::isa<mlir::LLVM::LLVMPointerType>(AddrV.getType())) {
    Ptr = AddrV;
  } else if (auto SlotTy = dyn_cast<MemRefType>(AddrV.getType());
             SlotTy && SlotTy.getRank() == 1 && SlotTy.getShape()[0] == 1 &&
             mlir::isa<mlir::LLVM::LLVMPointerType>(SlotTy.getElementType())) {
    mlir::Value Zero = createOp<arith::ConstantIndexOp>(PImpl->Builder, Loc, 0);
    Ptr =
        createOp<memref::LoadOp>(PImpl->Builder, Loc, AddrV, ValueRange{Zero});
  } else {
    reportFatalError("createAtomicMin: expected !llvm.ptr or "
                     "memref<1x!llvm.ptr> address slot");
  }

  mlir::Value V = PImpl->unwrap(Val);
  if (mlir::isa<mlir::FloatType>(V.getType()) &&
      !isSupportedAtomicFloatType(V.getType()))
    reportFatalError("createAtomicMin: unsupported floating-point atomic type");

  mlir::LLVM::AtomicBinOp BinOp;
  if (mlir::isa<mlir::FloatType>(V.getType()))
    BinOp = mlir::LLVM::AtomicBinOp::fmin;
  else
    BinOp = mlir::LLVM::AtomicBinOp::min;

  auto Atomic = createOp<mlir::LLVM::AtomicRMWOp>(
      PImpl->Builder, Loc, V.getType(), BinOp, Ptr, V,
      mlir::LLVM::AtomicOrdering::seq_cst,
      /*syncscope=*/mlir::StringAttr{}, /*alignment=*/mlir::IntegerAttr{},
      /*volatile_=*/false, /*access_groups=*/mlir::ArrayAttr{},
      /*alias_scopes=*/mlir::ArrayAttr{}, /*noalias_scopes=*/mlir::ArrayAttr{},
      /*tbaa=*/mlir::ArrayAttr{});
  return PImpl->wrap(Atomic.getResult());
}

IRValue *MLIRCodeBuilder::loadFromPointee(IRValue *Slot, IRType /*AllocTy*/,
                                          IRType ValueTy) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  auto LLVMPointerTy = mlir::LLVM::LLVMPointerType::get(&PImpl->Context);

  mlir::Value SlotV = PImpl->unwrap(Slot);
  auto SlotTy = dyn_cast<MemRefType>(SlotV.getType());
  if (!SlotTy || SlotTy.getRank() != 1 || SlotTy.getShape()[0] != 1 ||
      !mlir::isa<mlir::LLVM::LLVMPointerType>(SlotTy.getElementType()))
    reportFatalError(
        "loadFromPointee: expected address slot (memref<1x!llvm.ptr<...>>)");

  mlir::Value Zero = createOp<arith::ConstantIndexOp>(PImpl->Builder, Loc, 0);
  mlir::Value PtrVal =
      createOp<memref::LoadOp>(PImpl->Builder, Loc, SlotV, ValueRange{Zero});

  mlir::Type LoadTy = (ValueTy.Kind == IRTypeKind::Pointer)
                          ? mlir::Type(LLVMPointerTy)
                          : toMLIRScalarType(ValueTy.Kind, PImpl->Context);
  mlir::Value Loaded =
      createOp<mlir::LLVM::LoadOp>(PImpl->Builder, Loc, LoadTy, PtrVal);
  return PImpl->wrap(Loaded);
}

void MLIRCodeBuilder::storeToPointee(IRValue *Slot, IRType /*AllocTy*/,
                                     IRValue *Val) {
  auto Loc = PImpl->Builder.getUnknownLoc();

  mlir::Value SlotV = PImpl->unwrap(Slot);
  auto SlotTy = dyn_cast<MemRefType>(SlotV.getType());
  if (!SlotTy || SlotTy.getRank() != 1 || SlotTy.getShape()[0] != 1 ||
      !mlir::isa<mlir::LLVM::LLVMPointerType>(SlotTy.getElementType()))
    reportFatalError(
        "storeToPointee: expected address slot (memref<1x!llvm.ptr<...>>)");

  mlir::Value Zero = createOp<arith::ConstantIndexOp>(PImpl->Builder, Loc, 0);
  mlir::Value PtrVal =
      createOp<memref::LoadOp>(PImpl->Builder, Loc, SlotV, ValueRange{Zero});
  mlir::Value V = PImpl->unwrap(Val);
  createOp<mlir::LLVM::StoreOp>(PImpl->Builder, Loc, V, PtrVal);
}

VarAlloc MLIRCodeBuilder::allocPointer(const std::string & /*Name*/,
                                       IRType ElemTy, unsigned AddrSpace) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  auto LLVMPointerTy =
      mlir::LLVM::LLVMPointerType::get(&PImpl->Context, AddrSpace);
  auto PtrSlotTy = MemRefType::get({1}, LLVMPointerTy);
  mlir::Value PtrSlot;
  {
    OpBuilder::InsertionGuard Guard(PImpl->Builder);
    PImpl->Builder.setInsertionPointToStart(PImpl->EntryBlock);
    PtrSlot = createOp<memref::AllocaOp>(PImpl->Builder, Loc, PtrSlotTy);
  }
  IRType AllocTy{IRTypeKind::Pointer, ElemTy.Signed, 0, ElemTy.Kind};
  return {PImpl->wrap(PtrSlot), ElemTy, AllocTy, AddrSpace};
}

VarAlloc MLIRCodeBuilder::allocArray(const std::string &Name, AddressSpace AS,
                                     IRType ElemTy, size_t NElem) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  mlir::Type MLIRElemTy = toMLIRScalarType(ElemTy.Kind, PImpl->Context);

  mlir::Attribute MemorySpaceAttr;
  if (AS != AddressSpace::DEFAULT) {
    auto I64Ty = mlir::IntegerType::get(&PImpl->Context, 64);
    MemorySpaceAttr = mlir::IntegerAttr::get(I64Ty, static_cast<unsigned>(AS));
  }

  auto ArrMemRefTy =
      MemRefType::get({static_cast<int64_t>(NElem)}, MLIRElemTy,
                      /*layout=*/mlir::MemRefLayoutAttrInterface{},
                      /*memorySpace=*/MemorySpaceAttr);
  IRType AllocTy{IRTypeKind::Array, ElemTy.Signed, NElem, ElemTy.Kind};

  // In device mode, SHARED arrays must be backed by workgroup memory.
  // Model them as `memref.global` in addrspace(3) and use `memref.get_global`
  // inside kernels. This lowers to an internal `addrspace(3)` global in LLVM
  // IR.
  if (PImpl->TargetModel != TargetModelType::HOST &&
      AS == AddressSpace::SHARED) {
    if (!PImpl->DeviceModule)
      reportFatalError(
          "allocArray: expected active gpu.module for device code");

    // `memref.global` is a symbol; symbols must be unique within the gpu.module
    // symbol table. Auto-unique on collisions by appending a numeric suffix.
    std::string UniqueName = Name.empty() ? "allocArray" : Name;
    if (PImpl->DeviceModule.lookupSymbol(UniqueName) != nullptr) {
      unsigned UniquingCounter = 0;
      auto HasSymbolConflict = [&](llvm::StringRef Candidate) {
        return PImpl->DeviceModule.lookupSymbol(Candidate) != nullptr;
      };
      auto Unique = mlir::SymbolTable::generateSymbolName<64>(
          UniqueName, HasSymbolConflict, UniquingCounter);
      UniqueName.assign(Unique.begin(), Unique.end());
    }

    {
      OpBuilder::InsertionGuard Guard(PImpl->Builder);
      PImpl->Builder.setInsertionPointToStart(PImpl->DeviceModule.getBody());
      createOp<memref::GlobalOp>(
          PImpl->Builder, Loc, UniqueName,
          /*sym_visibility=*/PImpl->Builder.getStringAttr("private"),
          /*type=*/ArrMemRefTy,
          /*initial_value=*/mlir::UnitAttr::get(&PImpl->Context),
          /*constant=*/false,
          /*alignment=*/mlir::IntegerAttr{});
    }

    mlir::Value Global = createOp<memref::GetGlobalOp>(PImpl->Builder, Loc,
                                                       ArrMemRefTy, UniqueName);
    return {PImpl->wrap(Global), ElemTy, AllocTy, static_cast<unsigned>(AS)};
  }

  mlir::Value Alloca;
  {
    OpBuilder::InsertionGuard Guard(PImpl->Builder);
    PImpl->Builder.setInsertionPointToStart(PImpl->EntryBlock);
    Alloca = createOp<memref::AllocaOp>(PImpl->Builder, Loc, ArrMemRefTy);
  }
  return {PImpl->wrap(Alloca), ElemTy, AllocTy, static_cast<unsigned>(AS)};
}

#if defined(PROTEUS_ENABLE_CUDA) || defined(PROTEUS_ENABLE_HIP)
void MLIRCodeBuilder::setLaunchBoundsForKernel(IRFunction *F,
                                               int MaxThreadsPerBlock,
                                               int MinBlocksPerSM) {
  if (MaxThreadsPerBlock <= 0 || MinBlocksPerSM < 0)
    reportFatalError("MLIRCodeBuilder::setLaunchBoundsForKernel: invalid "
                     "launch-bounds arguments");

  auto *MF = PImpl->unwrapFunction(F);
  if (!MF->IsKernel)
    reportFatalError("MLIRCodeBuilder::setLaunchBoundsForKernel: function is "
                     "not marked as kernel");

  auto KernelFn = dyn_cast<gpu::GPUFuncOp>(MF->Op);
  if (!KernelFn)
    reportFatalError("MLIRCodeBuilder::setLaunchBoundsForKernel: expected "
                     "gpu.func kernel");

  if (TargetModel == TargetModelType::CUDA) {
    // Canonical GPU->NVVM lowering expects NVVM launch-bounds attributes on
    // the kernel function (nvvm.maxntid / nvvm.minctasm).
    KernelFn->setAttr("nvvm.maxntid", PImpl->Builder.getDenseI32ArrayAttr(
                                          {MaxThreadsPerBlock, 1, 1}));
    KernelFn->setAttr(
        "nvvm.minctasm",
        IntegerAttr::get(mlir::IntegerType::get(&PImpl->Context, 32),
                         MinBlocksPerSM));
    return;
  }

  if (TargetModel == TargetModelType::HIP) {
    // ROCDL launch bounds are represented with canonical rocdl.* attributes;
    // these map to AMDGPU kernel function attributes in LLVM lowering.
    KernelFn->setAttr(
        "rocdl.flat_work_group_size",
        StringAttr::get(&PImpl->Context,
                        "1," + std::to_string(MaxThreadsPerBlock)));
    KernelFn->setAttr(
        "rocdl.max_flat_work_group_size",
        IntegerAttr::get(mlir::IntegerType::get(&PImpl->Context, 32),
                         MaxThreadsPerBlock));
    KernelFn->setAttr(
        "rocdl.waves_per_eu",
        IntegerAttr::get(mlir::IntegerType::get(&PImpl->Context, 32),
                         MinBlocksPerSM));
    return;
  }

  reportFatalError("MLIRCodeBuilder::setLaunchBoundsForKernel: unsupported "
                   "target model for launch bounds");
}
#endif

} // namespace proteus
