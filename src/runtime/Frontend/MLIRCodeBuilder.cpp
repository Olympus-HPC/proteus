#include "proteus/Frontend/MLIRCodeBuilder.h"
#include "proteus/Error.h"
#include "proteus/Frontend/IRType.h"
#include "proteus/impl/CoreLLVM.h"
#include "proteus/impl/MLIRIRFunction.h"
#include "proteus/impl/MLIRIRValue.h"

#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/MathToLLVM/MathToLLVM.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/Passes.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#if PROTEUS_ENABLE_HIP
#include <mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h>
#endif
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#if PROTEUS_ENABLE_CUDA
#include <mlir/Dialect/LLVMIR/NVVMDialect.h>
#endif
#if PROTEUS_ENABLE_HIP
#include <mlir/Dialect/LLVMIR/ROCDLDialect.h>
#endif
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVM/ModuleToObject.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#if PROTEUS_ENABLE_CUDA
#include <mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h>
#endif
#if PROTEUS_ENABLE_HIP
#include <mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h>
#endif
#include <mlir/Target/LLVMIR/Export.h>

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBufferRef.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>

#include <deque>
#include <mutex>
#include <optional>
#include <unordered_map>

namespace proteus {

using namespace mlir;

static void ensureLLVMTargetsInitialized() {
  static InitLLVMTargets Init;
  (void)Init;
}

static std::string computeTargetDataLayout(llvm::StringRef TargetTriple,
                                           llvm::StringRef CPU,
                                           llvm::StringRef Features) {
  ensureLLVMTargetsInitialized();

  static std::mutex CacheMu;
  static std::unordered_map<std::string, std::string> Cache;

  const std::string Key =
      (TargetTriple.str() + "\n" + CPU.str() + "\n" + Features.str());
  {
    std::lock_guard<std::mutex> Guard(CacheMu);
    if (auto It = Cache.find(Key); It != Cache.end())
      return It->second;
  }

  std::string Error;
  const llvm::Target *T =
      llvm::TargetRegistry::lookupTarget(TargetTriple, Error);
  if (!T)
    reportFatalError("MLIRCodeBuilder: failed to lookup LLVM target for '" +
                     TargetTriple.str() + "': " + Error);

  llvm::TargetOptions Options;
  std::optional<llvm::Reloc::Model> RelocModel;
  std::optional<llvm::CodeModel::Model> CodeModel;

  std::unique_ptr<llvm::TargetMachine> TM(T->createTargetMachine(
      TargetTriple, CPU, Features, Options, RelocModel, CodeModel,
      /*OptLevel=*/llvm::CodeGenOptLevel::Default));
  if (!TM)
    reportFatalError("MLIRCodeBuilder: failed to create LLVM TargetMachine for "
                     "'" +
                     TargetTriple.str() + "'");

  std::string DataLayout = TM->createDataLayout().getStringRepresentation();
  {
    std::lock_guard<std::mutex> Guard(CacheMu);
    auto [It, Inserted] = Cache.emplace(Key, std::move(DataLayout));
    (void)Inserted;
    return It->second;
  }
}

static bool hasLLVMFunctionDefinitions(llvm::Module &Mod) {
  for (auto &Fn : Mod.functions())
    if (!Fn.isDeclaration())
      return true;
  return false;
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

// Device kernels lower with `useBarePtrCallConv=true` (HIP) and require memrefs
// in the kernel ABI to have static shape and identity layout. For pointer-like
// arguments, use a static 1-D memref as a "pointer proxy" that can be lowered
// to a raw pointer in the backend ABI.
static mlir::Type toDeviceMLIRType(IRType Ty, MLIRContext &Ctx) {
  if (Ty.Kind == IRTypeKind::Pointer) {
    mlir::Type ElemTy = toMLIRScalarType(Ty.ElemKind, Ctx);
    return MemRefType::get({1}, ElemTy);
  }
  return toMLIRType(Ty, Ctx);
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
  gpu::GPUModuleOp DeviceModule;

  // Optional device architecture string used during device lowering.
  // For HIP this is the ROCDL chipset (e.g. gfx90a). For CUDA it is unused.
  std::string DeviceArch;

  // Lowering is performed lazily when JitModule requests compilation
  // artifacts.
  std::unique_ptr<llvm::LLVMContext> LLVMCtx;
  std::unique_ptr<llvm::Module> LLVMMod;
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

  // Side table for pointer (base memref, offset slot) representation.
  struct PointerInfo {
    mlir::Value BaseMemRef;
  };
  llvm::DenseMap<mlir::Value, PointerInfo> PointerMap;

  enum class AtomicOp { Add, Sub, Max, Min };

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

  explicit Impl() : Builder(&Context) {
    Context.loadDialect<
        mlir::func::FuncDialect, arith::ArithDialect, cf::ControlFlowDialect,
        gpu::GPUDialect, LLVM::LLVMDialect,
#if PROTEUS_ENABLE_CUDA
        NVVM::NVVMDialect,
#endif
#if PROTEUS_ENABLE_HIP
        ROCDL::ROCDLDialect,
#endif
        math::MathDialect, memref::MemRefDialect, scf::SCFDialect>();
    Module = ModuleOp::create(Builder.getUnknownLoc());
  }

  void lowerToLLVM(TargetModelType TM, int OptLevel,
                   llvm::StringRef TargetTriple, llvm::StringRef CPU,
                   llvm::StringRef Features) {
    if (LoweredToLLVM)
      return;

    if (failed(mlir::verify(Module)))
      reportFatalError("MLIRCodeBuilder: invalid MLIR module before lowering");

    mlir::PassManager PM(&Context);
    auto SetDeviceModuleTargetAttrs = [&]() {
      const std::string DataLayout =
          computeTargetDataLayout(TargetTriple, CPU, Features);
      Module.getOperation()->setAttr(
          mlir::LLVM::LLVMDialect::getTargetTripleAttrName(),
          Builder.getStringAttr(TargetTriple));
      Module.getOperation()->setAttr(
          mlir::LLVM::LLVMDialect::getDataLayoutAttrName(),
          Builder.getStringAttr(DataLayout));
    };
    auto AddCommonGpuLoweringPrelude = [&]() {
      PM.addPass(mlir::createConvertSCFToCFPass());
      {
        mlir::ConvertIndexToLLVMPassOptions Opts;
        Opts.indexBitwidth = mlir::kDeriveIndexBitwidthFromDataLayout;
        PM.addPass(mlir::createConvertIndexToLLVMPass(Opts));
      }
      {
        mlir::ConvertControlFlowToLLVMPassOptions Opts;
        Opts.indexBitwidth = mlir::kDeriveIndexBitwidthFromDataLayout;
        PM.addPass(mlir::createConvertControlFlowToLLVMPass(Opts));
      }
      PM.addPass(mlir::createReconcileUnrealizedCastsPass());
    };

    if (TM == TargetModelType::HOST) {
      // HOST path: input is host-side structured IR (func/scf/arith/math/
      // memref); output is host-targeted LLVM dialect before translation.
      PM.addPass(mlir::createConvertSCFToCFPass());
      PM.addPass(mlir::createArithToLLVMConversionPass());
      PM.addPass(mlir::createConvertMathToLLVMPass());
      PM.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
      PM.addPass(mlir::createConvertFuncToLLVMPass());
      PM.addPass(mlir::createReconcileUnrealizedCastsPass());
    } else if (TM == TargetModelType::CUDA) {
#if PROTEUS_ENABLE_CUDA
      // CUDA path: lower device-only gpu.module/gpu.func to NVVM/LLVM dialect.
      // This pass is defined on gpu.module, so schedule it as a nested pass.
      mlir::ConvertGpuOpsToNVVMOpsOptions NVVMOpts;
      NVVMOpts.indexBitwidth = mlir::kDeriveIndexBitwidthFromDataLayout;
      // Use bare pointer call convention so kernel argument ABI matches the
      // existing Proteus CUDA dispatcher launch path (raw pointer parameters).
      NVVMOpts.useBarePtrCallConv = true;
      // Provide a target triple + data layout so index bitwidth can be derived
      // consistently during conversion.
      SetDeviceModuleTargetAttrs();

      // NOTE: ConvertControlFlowToLLVMPass / ConvertIndexToLLVMPass are
      // restricted to `builtin.module`, so they cannot be added to a nested
      // `gpu.module` pass manager. Run them at the top-level before GPU->NVVM
      // conversion creates llvm.func bodies.
      AddCommonGpuLoweringPrelude();

      PM.nest<gpu::GPUModuleOp>().addPass(
          mlir::createConvertGpuOpsToNVVMOps(NVVMOpts));
#else
      reportFatalError(
          "MLIRCodeBuilder: CUDA target requested but Proteus was built with "
          "PROTEUS_ENABLE_CUDA=OFF");
#endif
    } else if (TM == TargetModelType::HIP) {
#if PROTEUS_ENABLE_HIP
      // HIP path: input is device-only gpu.module/gpu.func + gpu.* ops;
      // output is ROCDL/LLVM dialect that can be translated to AMDGPU LLVM IR.
      // Step 1 (gpu -> rocdl/llvm): lower gpu.module/gpu.func and gpu.* ops.
      // This pass is defined on gpu.module, so schedule it as a nested pass
      // instead of directly on builtin.module.
      // Use bare pointer call convention so kernel argument ABI matches the
      // existing Proteus HIP dispatcher launch path (raw pointer parameters).
      // The ROCDL lowering pass requires a non-empty chipset name.
      if (CPU.empty())
        reportFatalError(
            "MLIRCodeBuilder: HIP device lowering requires a "
            "non-empty device arch (call setDeviceArch(\"gfx*\"))");
      const std::string HipChipset = CPU.str();
      // Provide a target triple + data layout so index bitwidth can be derived
      // consistently during conversion.
      SetDeviceModuleTargetAttrs();

      // NOTE: ConvertControlFlowToLLVMPass / ConvertIndexToLLVMPass are
      // restricted to `builtin.module`, so they cannot be added to a nested
      // `gpu.module` pass manager. Run them at the top-level before GPU->ROCDL
      // conversion creates llvm.func bodies.
      AddCommonGpuLoweringPrelude();

      PM.nest<gpu::GPUModuleOp>().addPass(mlir::createLowerGpuOpsToROCDLOpsPass(
          /*chipset=*/HipChipset,
          /*indexBitwidth=*/
          mlir::kDeriveIndexBitwidthFromDataLayout,
          /*useBarePtrCallConv=*/true));
#else
      reportFatalError(
          "MLIRCodeBuilder: HIP target requested but Proteus was built with "
          "PROTEUS_ENABLE_HIP=OFF");
#endif
    } else {
      reportFatalError(
          "MLIRCodeBuilder: lowering only supported for HOST/CUDA/HIP");
    }

    if (failed(PM.run(Module)))
      reportFatalError("MLIRCodeBuilder: failed to lower MLIR module to LLVM "
                       "dialect");

    LLVMCtx = std::make_unique<llvm::LLVMContext>();

    // Register dialect translations used by translateModuleToLLVMIR.
    mlir::registerBuiltinDialectTranslation(Context);
    mlir::registerLLVMDialectTranslation(Context);
    if (TM == TargetModelType::CUDA || TM == TargetModelType::HIP)
      mlir::registerGPUDialectTranslation(Context);
    if (TM == TargetModelType::CUDA) {
#if PROTEUS_ENABLE_CUDA
      mlir::registerNVVMDialectTranslation(Context);
#else
      reportFatalError(
          "MLIRCodeBuilder: CUDA target requested but Proteus was built with "
          "PROTEUS_ENABLE_CUDA=OFF");
#endif
    }
    if (TM == TargetModelType::HIP) {
#if PROTEUS_ENABLE_HIP
      mlir::registerROCDLDialectTranslation(Context);
#else
      reportFatalError(
          "MLIRCodeBuilder: HIP target requested but Proteus was built with "
          "PROTEUS_ENABLE_HIP=OFF");
#endif
    }

    Operation *OperationToSerialize = Module.getOperation();
    if (TM == TargetModelType::CUDA || TM == TargetModelType::HIP) {
      auto DeviceSym = Module.lookupSymbol<gpu::GPUModuleOp>("kernels");
      if (!DeviceSym)
        reportFatalError("MLIRCodeBuilder: expected gpu.module @kernels for "
                         "device lowering");
      mlir::PassManager DevicePM(&Context);
      DevicePM.nest<gpu::GPUModuleOp>().addPass(
          mlir::createReconcileUnrealizedCastsPass());
      if (failed(DevicePM.run(Module)))
        reportFatalError("MLIRCodeBuilder: failed to reconcile device "
                         "gpu.module after lowering");

      DeviceSym = Module.lookupSymbol<gpu::GPUModuleOp>("kernels");
      if (!DeviceSym)
        reportFatalError("MLIRCodeBuilder: reconciled device module lost "
                         "gpu.module @kernels");

      bool HasUnrealizedCasts = false;
      DeviceSym.walk(
          [&](mlir::UnrealizedConversionCastOp) { HasUnrealizedCasts = true; });
      if (HasUnrealizedCasts) {
        llvm::errs()
            << "[proteus][MLIRCodeBuilder] Lowered device gpu.module still "
               "contains builtin.unrealized_conversion_cast:\n";
        DeviceSym.print(llvm::errs());
        llvm::errs() << "\n";
        reportFatalError("MLIRCodeBuilder: device gpu.module still contains "
                         "unrealized conversion casts after reconciliation");
      }

      OperationToSerialize = DeviceSym.getOperation();
    }

    if (TM == TargetModelType::CUDA || TM == TargetModelType::HIP) {
      mlir::LLVM::ModuleToObject Serializer(*OperationToSerialize, TargetTriple,
                                            CPU, Features, OptLevel);
      std::optional<SmallVector<char, 0>> Serialized = Serializer.run();
      if (!Serialized)
        reportFatalError("MLIRCodeBuilder: failed to serialize device module "
                         "to LLVM bitcode");

      llvm::MemoryBufferRef Buffer(
          StringRef(Serialized->data(), Serialized->size()), "DeviceBitcode");
      auto ParsedModule = llvm::parseBitcodeFile(Buffer, *LLVMCtx);
      if (!ParsedModule)
        reportFatalError("MLIRCodeBuilder: failed to parse serialized device "
                         "LLVM bitcode: " +
                         llvm::toString(ParsedModule.takeError()));
      LLVMMod = std::move(*ParsedModule);
    } else {
      LLVMMod = mlir::translateModuleToLLVMIR(Module.getOperation(), *LLVMCtx);
      if (!LLVMMod)
        reportFatalError("MLIRCodeBuilder: failed to translate LLVM dialect "
                         "to LLVM IR module");
    }

    if (TM == TargetModelType::CUDA || TM == TargetModelType::HIP) {
      if (!hasLLVMFunctionDefinitions(*LLVMMod)) {
        llvm::errs()
            << "[proteus][MLIRCodeBuilder] Device translation produced no "
               "function definitions. MLIR device module was:\n";
        OperationToSerialize->print(llvm::errs());
        llvm::errs() << "\n";
        reportFatalError("MLIRCodeBuilder: device translation produced an "
                         "empty LLVM IR module");
      }
    }

    // MLIR device lowering may qualify kernel function symbols (e.g. with the
    // gpu.module name). Normalize known prefixes so the rest of Proteus can
    // resolve functions by their frontend names.
    if (TM == TargetModelType::CUDA || TM == TargetModelType::HIP) {
      llvm::SmallVector<std::pair<llvm::Function *, std::string>> ToRename;
      llvm::StringSet<> NewNames;

      for (auto &Fn : LLVMMod->functions()) {
        if (Fn.isDeclaration())
          continue;

        llvm::StringRef Name = Fn.getName();
        static constexpr llvm::StringLiteral KernelsPrefix = "kernels.";
        if (!Name.starts_with(KernelsPrefix))
          continue;

        llvm::StringRef Stripped = Name.drop_front(KernelsPrefix.size());
        if (Stripped.empty())
          continue;

        if (LLVMMod->getFunction(Stripped))
          reportFatalError(
              "MLIRCodeBuilder: device symbol name collision for " +
              Stripped.str());

        if (!NewNames.insert(Stripped).second)
          reportFatalError("MLIRCodeBuilder: multiple device symbols map to " +
                           Stripped.str());

        ToRename.emplace_back(&Fn, Stripped.str());
      }

      for (auto &[Fn, NewName] : ToRename)
        Fn->setName(NewName);
    }

    // Keep these knobs in place so call sites can pass target options when
    // needed; the current host path relies on module defaults. For device
    // targets, also set the data layout so the downstream device JIT sees a
    // consistent module configuration.
    (void)OptLevel;
    if (!TargetTriple.empty()) {
      LLVMMod->setTargetTriple(TargetTriple.str());
      if (TM == TargetModelType::CUDA || TM == TargetModelType::HIP) {
        LLVMMod->setDataLayout(
            computeTargetDataLayout(TargetTriple, CPU, Features));
      }
    }

    LoweredToLLVM = true;
  }

  IRValue *wrap(mlir::Value V) {
    Values.emplace_back(V);
    return &Values.back();
  }

  mlir::Value unwrap(IRValue *V) { return static_cast<MLIRIRValue *>(V)->V; }

  IRFunction *wrapFunction(mlir::Operation *Op, bool IsKernel) {
    Functions.emplace_back(Op, IsKernel);
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
    DeviceModule = Builder.create<gpu::GPUModuleOp>(Builder.getUnknownLoc(),
                                                    DeviceModuleName);
    return DeviceModule;
  }

  std::pair<mlir::Value, mlir::Value> resolveAtomicAddress(IRValue *Addr) {
    mlir::Value Slot = unwrap(Addr);
    auto It = PointerMap.find(Slot);
    if (It == PointerMap.end())
      reportFatalError("atomic on non-pointer address");

    mlir::Value Base = It->second.BaseMemRef;
    if (!Base)
      reportFatalError("atomic on non-pointer address");

    auto SlotTy = dyn_cast<MemRefType>(Slot.getType());
    if (!SlotTy || SlotTy.getRank() != 1 || SlotTy.getShape()[0] != 1 ||
        !SlotTy.getElementType().isIndex())
      reportFatalError("atomic on non-pointer address");

    auto Loc = Builder.getUnknownLoc();
    mlir::Value Zero = Builder.create<arith::ConstantIndexOp>(Loc, 0);
    mlir::Value Idx =
        Builder.create<memref::LoadOp>(Loc, Slot, ValueRange{Zero});
    return {Base, Idx};
  }

  std::pair<mlir::Value, mlir::Value> resolvePointerValue(IRValue *Ptr) {
    // Pointer values in the MLIR backend are represented as:
    //   - slot: memref<1xindex> holding the current offset at [0]
    //   - PointerMap[slot]: side-table entry holding the base memref
    // This helper validates the representation and returns {slot, base}.
    mlir::Value Slot = unwrap(Ptr);
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
        Builder.create<func::FuncOp>(Builder.getUnknownLoc(), Name, FTy);
    NewFunc.setSymVisibilityAttr(Builder.getStringAttr("private"));
    return NewFunc;
  }

  gpu::GPUFuncOp lookupDeviceFunc(StringRef Name) {
    if (!DeviceModule)
      return gpu::GPUFuncOp{};
    return DeviceModule.lookupSymbol<gpu::GPUFuncOp>(Name);
  }

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
    mlir::Value Zero = Builder.create<arith::ConstantIndexOp>(Loc, 0);
    mlir::Value Idx =
        Builder.create<memref::LoadOp>(Loc, PtrSlot, ValueRange{Zero});
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

    mlir::Type ElemTy = BaseTy.getElementType();
    if (Val.getType() != ElemTy)
      reportFatalError("atomic op not supported for this type");

    const bool IsInt = mlir::isa<mlir::IntegerType>(ElemTy);
    const bool IsFloat = mlir::isa<mlir::FloatType>(ElemTy);
    if (!IsInt && !IsFloat)
      reportFatalError("atomic op not supported for this type");
    if (IsFloat && !isSupportedAtomicFloatType(ElemTy))
      reportFatalError("atomic op not supported for this type");

    const char *KindStr = nullptr;
    if (mlir::isa<mlir::IntegerType>(ElemTy)) {
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
    } else if (mlir::isa<mlir::FloatType>(ElemTy)) {
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

      mlir::Value Cur = Body.getArgument(0);
      mlir::Value New;
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

void MLIRCodeBuilder::printLLVMIR(llvm::raw_ostream &OS) {
  ensureLoweredToLLVM(/*OptLevel=*/3, /*TargetTriple=*/"", /*Features=*/"");
  if (!PImpl->LLVMMod)
    reportFatalError("MLIRCodeBuilder::printLLVMIR: LLVM module ownership was "
                     "already transferred");
  PImpl->LLVMMod->print(OS, nullptr);
}

void MLIRCodeBuilder::ensureLoweredToLLVM(int OptLevel,
                                          const std::string &TargetTriple,
                                          const std::string &Features) {
  const std::string EffectiveTriple =
      TargetTriple.empty() ? getTargetTriple(TargetModel) : TargetTriple;
  PImpl->lowerToLLVM(TargetModel, OptLevel, EffectiveTriple,
                     llvm::StringRef{PImpl->DeviceArch}, Features);
}

std::unique_ptr<llvm::LLVMContext> MLIRCodeBuilder::takeContext() {
  ensureLoweredToLLVM(/*OptLevel=*/3, /*TargetTriple=*/"", /*Features=*/"");
  return std::move(PImpl->LLVMCtx);
}

std::unique_ptr<llvm::Module> MLIRCodeBuilder::takeModule() {
  ensureLoweredToLLVM(/*OptLevel=*/3, /*TargetTriple=*/"", /*Features=*/"");
  return std::move(PImpl->LLVMMod);
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
  auto MapIRType = [&](IRType Ty) -> mlir::Type {
    return IsDeviceMode ? toDeviceMLIRType(Ty, Ctx) : toMLIRType(Ty, Ctx);
  };

  llvm::SmallVector<mlir::Type> MLIRArgTys;
  MLIRArgTys.reserve(ArgTys.size());
  for (const auto &AT : ArgTys)
    MLIRArgTys.push_back(MapIRType(AT));

  llvm::SmallVector<mlir::Type> RetTys;
  if (RetTy.Kind != IRTypeKind::Void)
    RetTys.push_back(MapIRType(RetTy));

  auto FTy = mlir::FunctionType::get(&Ctx, MLIRArgTys, RetTys);
  auto Loc = PImpl->Builder.getUnknownLoc();

  if (IsDeviceMode) {
    auto DeviceModule = PImpl->getOrCreateDeviceModule();
    if (DeviceModule.getBodyRegion().empty())
      DeviceModule.getBodyRegion().push_back(new Block());
    OpBuilder::InsertionGuard Guard(PImpl->Builder);
    Block &DeviceBody = DeviceModule.getBodyRegion().front();
    // gpu.module owns a terminator (gpu.module_end); insert new gpu.func
    // declarations before the terminator to keep the IR structurally valid.
    if (!DeviceBody.empty() &&
        DeviceBody.back().hasTrait<OpTrait::IsTerminator>()) {
      PImpl->Builder.setInsertionPoint(&DeviceBody.back());
    } else {
      PImpl->Builder.setInsertionPointToEnd(&DeviceBody);
    }

    // In CUDA/HIP mode Proteus emits device-side IR under gpu.module;
    // host-side orchestration remains outside MLIR. Device helpers are
    // regular gpu.func without the gpu.kernel marker.
    // Keep gpu.func symbol names equal to frontend names so the existing
    // Proteus hash-based mangling step can find and rename the LLVM function.
    auto KernelOp = PImpl->Builder.create<gpu::GPUFuncOp>(Loc, Name, FTy);
    if (KernelOp.getBody().empty())
      KernelOp.addEntryBlock();
    if (IsKernel)
      // Keep kernel marker explicit so downstream GPU passes can identify
      // launchable entry points before target-specific lowering.
      KernelOp->setAttr("gpu.kernel", UnitAttr::get(&PImpl->Context));

    PImpl->CurrentFuncOp = KernelOp.getOperation();
    PImpl->CurrentIsKernel = IsKernel;
    PImpl->EntryBlock = &KernelOp.getBody().front();
    return PImpl->wrapFunction(KernelOp.getOperation(), IsKernel);
  }

  if (IsKernel)
    reportFatalError("MLIRCodeBuilder::addFunction: host target does not "
                     "support kernel functions");

  // Host mode keeps plain func.func definitions in the top-level module.
  PImpl->Builder.setInsertionPointToEnd(PImpl->Module.getBody());
  auto FuncOp = PImpl->Builder.create<mlir::func::FuncOp>(Loc, Name, FTy);
  FuncOp.addEntryBlock();

  PImpl->CurrentFuncOp = FuncOp.getOperation();
  PImpl->CurrentIsKernel = false;
  PImpl->EntryBlock = &FuncOp.getBody().front();
  return PImpl->wrapFunction(FuncOp.getOperation(), /*IsKernel=*/false);
}

// ---------------------------------------------------------------------------
// setFunctionName
// ---------------------------------------------------------------------------

void MLIRCodeBuilder::setFunctionName(IRFunction *F, const std::string &Name) {
  auto *MF = PImpl->unwrapFunction(F);
  if (auto HostFn = dyn_cast<mlir::func::FuncOp>(MF->Op)) {
    HostFn.setName(Name);
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
  if (auto HostFn = dyn_cast<mlir::func::FuncOp>(MF->Op)) {
    return PImpl->wrap(HostFn.getBody().front().getArgument(Idx));
  }
  if (auto KernelFn = dyn_cast<gpu::GPUFuncOp>(MF->Op)) {
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
  if (auto HostFn = dyn_cast<mlir::func::FuncOp>(MF->Op)) {
    PImpl->EntryBlock = &HostFn.getBody().front();
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
      PImpl->Builder.create<gpu::ReturnOp>(Loc);
    else
      PImpl->Builder.create<mlir::func::ReturnOp>(Loc);
  }
  PImpl->CurrentFuncOp = nullptr;
  PImpl->CurrentIsKernel = false;
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
  if (PImpl->CurrentIsKernel)
    PImpl->Builder.create<gpu::ReturnOp>(Loc);
  else
    PImpl->Builder.create<mlir::func::ReturnOp>(Loc);
}

void MLIRCodeBuilder::createRet(IRValue *V) {
  if (PImpl->CurrentIsKernel)
    reportFatalError("MLIRCodeBuilder::createRet: kernels must return void");
  auto Loc = PImpl->Builder.getUnknownLoc();
  PImpl->Builder.create<mlir::func::ReturnOp>(Loc, PImpl->unwrap(V));
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
  mlir::Value Src = PImpl->unwrap(V);
  mlir::Type DstTy = toMLIRType(ToTy, PImpl->Context);
  mlir::Value Result;

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
    else if (ToBits == FromBits)
      Result = Src;
    else if (FromTy.Signed)
      Result = PImpl->Builder.create<arith::ExtSIOp>(Loc, DstTy, Src);
    else
      Result = PImpl->Builder.create<arith::ExtUIOp>(Loc, DstTy, Src);
  } else if (FromFloat && ToFloat) {
    const unsigned FromBits = Src.getType().getIntOrFloatBitWidth();
    const unsigned ToBits = DstTy.getIntOrFloatBitWidth();
    if (ToBits < FromBits)
      Result = PImpl->Builder.create<arith::TruncFOp>(Loc, DstTy, Src);
    else if (ToBits == FromBits)
      Result = Src;
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
  mlir::Value C = PImpl->Builder.create<arith::ConstantOp>(Loc, Attr);
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
  mlir::Value C = PImpl->Builder.create<arith::ConstantOp>(Loc, Attr);
  return PImpl->wrap(C);
}

// ---------------------------------------------------------------------------
// loadScalar / storeScalar / allocScalar
// ---------------------------------------------------------------------------

IRValue *MLIRCodeBuilder::loadScalar(IRValue *Slot, IRType /*ValueTy*/) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  mlir::Value MemRefVal = PImpl->unwrap(Slot);
  mlir::Value Idx = PImpl->Builder.create<arith::ConstantIndexOp>(Loc, 0);
  mlir::Value Val =
      PImpl->Builder.create<memref::LoadOp>(Loc, MemRefVal, ValueRange{Idx});
  return PImpl->wrap(Val);
}

void MLIRCodeBuilder::storeScalar(IRValue *Slot, IRValue *Val) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  mlir::Value MemRefVal = PImpl->unwrap(Slot);
  mlir::Value V = PImpl->unwrap(Val);
  mlir::Value Idx = PImpl->Builder.create<arith::ConstantIndexOp>(Loc, 0);
  PImpl->Builder.create<memref::StoreOp>(Loc, V, MemRefVal, ValueRange{Idx});
}

VarAlloc MLIRCodeBuilder::allocScalar(const std::string & /*Name*/,
                                      IRType ValueTy) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  mlir::Type ElemTy = toMLIRType(ValueTy, PImpl->Context);
  // Allocate memref<1xT> — mirrors LLVM's alloca pattern.
  auto MemRefTy = MemRefType::get({1}, ElemTy);

  mlir::Value Alloca;
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
  mlir::Value CondV = PImpl->unwrap(Cond);

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
  mlir::Value LB = PImpl->Builder.create<arith::IndexCastOp>(
      Loc, PImpl->Builder.getIndexType(), PImpl->unwrap(InitVal));
  mlir::Value UB = PImpl->Builder.create<arith::IndexCastOp>(
      Loc, PImpl->Builder.getIndexType(), PImpl->unwrap(UpperBoundVal));
  mlir::Value Step = PImpl->Builder.create<arith::IndexCastOp>(
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
  mlir::Value IV = ForOp.getInductionVar();
  mlir::Type IterMLIRTy = toMLIRType(IterTy, PImpl->Context);
  mlir::Value TypedIV =
      PImpl->Builder.create<arith::IndexCastOp>(Loc, IterMLIRTy, IV);
  mlir::Value Idx = PImpl->Builder.create<arith::ConstantIndexOp>(Loc, 0);
  mlir::Value SlotV = PImpl->unwrap(IterSlot);
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
  mlir::Value CondV = PImpl->unwrap(CondIRV);

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
  mlir::Value Result = PImpl->Builder.create<arith::AndIOp>(
      Loc, PImpl->unwrap(LHS), PImpl->unwrap(RHS));
  return PImpl->wrap(Result);
}

IRValue *MLIRCodeBuilder::createOr(IRValue *LHS, IRValue *RHS) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  mlir::Value Result = PImpl->Builder.create<arith::OrIOp>(
      Loc, PImpl->unwrap(LHS), PImpl->unwrap(RHS));
  return PImpl->wrap(Result);
}

IRValue *MLIRCodeBuilder::createXor(IRValue *LHS, IRValue *RHS) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  mlir::Value Result = PImpl->Builder.create<arith::XOrIOp>(
      Loc, PImpl->unwrap(LHS), PImpl->unwrap(RHS));
  return PImpl->wrap(Result);
}

IRValue *MLIRCodeBuilder::createNot(IRValue *Val) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  // arith.xori %val, %true  where %true is arith.constant 1 : i1
  auto TrueAttr =
      IntegerAttr::get(mlir::IntegerType::get(&PImpl->Context, 1), 1);
  mlir::Value TrueVal = PImpl->Builder.create<arith::ConstantOp>(Loc, TrueAttr);
  mlir::Value Result =
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
    mlir::Value Val =
        PImpl->Builder.create<memref::LoadOp>(Loc, Base, ValueRange{Idx});
    if (Val.getType() != ExpectedTy)
      reportFatalError("MLIRCodeBuilder::createLoad: type mismatch for pointer "
                       "dereference load");
    return PImpl->wrap(Val);
  }

  // Scalar-slot case: Ptr is a mutable scalar slot represented as
  // memref<1xT>; load slot[0].
  if (Impl::isScalarSlotType(PtrV.getType())) {
    mlir::Value Zero = PImpl->Builder.create<arith::ConstantIndexOp>(Loc, 0);
    mlir::Value Val =
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
    mlir::Value Zero = PImpl->Builder.create<arith::ConstantIndexOp>(Loc, 0);
    PImpl->Builder.create<memref::StoreOp>(Loc, V, PtrV, ValueRange{Zero});
    return;
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

  mlir::Value Res = PImpl->Builder.create<arith::BitcastOp>(Loc, DstTy, Src);
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
    mlir::Value Res = PImpl->Builder.create<arith::ExtUIOp>(Loc, DstTy, Src);
    return PImpl->wrap(Res);
  }

  if ((SrcIsIndex && DstIsInt) || (SrcIsInt && DstIsIndex)) {
    mlir::Value Res =
        PImpl->Builder.create<arith::IndexCastUIOp>(Loc, DstTy, Src);
    return PImpl->wrap(Res);
  }

  reportFatalError("createZExt: unsupported type combination");
}
VarAlloc MLIRCodeBuilder::getElementPtr(IRValue *Base, IRType /*BaseTy*/,
                                        IRValue *Index, IRType ElemTy) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  mlir::Value IdxV = PImpl->unwrap(Index);

  // Cast index to index type if needed.
  if (!mlir::isa<mlir::IndexType>(IdxV.getType()))
    IdxV = PImpl->Builder.create<arith::IndexCastOp>(
        Loc, PImpl->Builder.getIndexType(), IdxV);

  // Allocate offset slot at entry block.
  auto IdxMemRefTy = MemRefType::get({1}, PImpl->Builder.getIndexType());
  mlir::Value OffsetSlot;
  {
    OpBuilder::InsertionGuard Guard(PImpl->Builder);
    PImpl->Builder.setInsertionPointToStart(PImpl->EntryBlock);
    OffsetSlot = PImpl->Builder.create<memref::AllocaOp>(Loc, IdxMemRefTy);
  }

  mlir::Value BaseMemRef;
  mlir::Value OffsetToStore;
  mlir::Value BaseV = PImpl->unwrap(Base);
  if (auto It = PImpl->PointerMap.find(BaseV); It != PImpl->PointerMap.end()) {
    BaseMemRef = It->second.BaseMemRef;
    if (!BaseMemRef)
      reportFatalError("getElementPtr: null pointer base");

    // Compose GEP offsets when Base is itself a pointer slot.
    mlir::Value BaseSlotV = BaseV;
    mlir::Value Idx0 = PImpl->Builder.create<arith::ConstantIndexOp>(Loc, 0);
    mlir::Value BaseOffset =
        PImpl->Builder.create<memref::LoadOp>(Loc, BaseSlotV, ValueRange{Idx0});
    OffsetToStore = PImpl->Builder.create<arith::AddIOp>(Loc, BaseOffset, IdxV);
  } else {
    BaseMemRef = PImpl->unwrap(Base);
    OffsetToStore = IdxV;
  }

  // Store the index as offset.
  mlir::Value Idx0 = PImpl->Builder.create<arith::ConstantIndexOp>(Loc, 0);
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
  mlir::Value IdxV =
      PImpl->Builder.create<arith::ConstantIndexOp>(Loc, (int64_t)Index);

  // Allocate offset slot at entry block.
  auto IdxMemRefTy = MemRefType::get({1}, PImpl->Builder.getIndexType());
  mlir::Value OffsetSlot;
  {
    OpBuilder::InsertionGuard Guard(PImpl->Builder);
    PImpl->Builder.setInsertionPointToStart(PImpl->EntryBlock);
    OffsetSlot = PImpl->Builder.create<memref::AllocaOp>(Loc, IdxMemRefTy);
  }

  mlir::Value BaseMemRef;
  mlir::Value OffsetToStore;
  mlir::Value BaseV = PImpl->unwrap(Base);
  if (auto It = PImpl->PointerMap.find(BaseV); It != PImpl->PointerMap.end()) {
    BaseMemRef = It->second.BaseMemRef;
    if (!BaseMemRef)
      reportFatalError("getElementPtr: null pointer base");

    // Compose GEP offsets when Base is itself a pointer slot.
    mlir::Value BaseSlotV = BaseV;
    mlir::Value Idx0 = PImpl->Builder.create<arith::ConstantIndexOp>(Loc, 0);
    mlir::Value BaseOffset =
        PImpl->Builder.create<memref::LoadOp>(Loc, BaseSlotV, ValueRange{Idx0});
    OffsetToStore = PImpl->Builder.create<arith::AddIOp>(Loc, BaseOffset, IdxV);
  } else {
    BaseMemRef = PImpl->unwrap(Base);
    OffsetToStore = IdxV;
  }

  // Store the index as offset.
  mlir::Value Idx0 = PImpl->Builder.create<arith::ConstantIndexOp>(Loc, 0);
  PImpl->Builder.create<memref::StoreOp>(Loc, OffsetToStore, OffsetSlot,
                                         ValueRange{Idx0});

  IRValue *SlotIRV = PImpl->wrap(OffsetSlot);

  PImpl->PointerMap[OffsetSlot] = {BaseMemRef};

  IRType AllocTy{IRTypeKind::Pointer, ElemTy.Signed, 0, ElemTy.Kind};
  return {SlotIRV, ElemTy, AllocTy, 0};
}
IRValue *MLIRCodeBuilder::createCall(const std::string &FName, IRType RetTy,
                                     const std::vector<IRType> &ArgTys,
                                     const std::vector<IRValue *> &Args) {
  if (ArgTys.size() != Args.size())
    reportFatalError("createCall: ArgTys.size() must match Args.size()");

  if (PImpl->isRawPointerAbiType(RetTy))
    reportFatalError("createCall: raw-pointer ABI calls are unsupported in "
                     "MLIR backend (memref-only). Change extern to take "
                     "memref or add LLVM-dialect lowering.");
  for (const auto &ArgTy : ArgTys) {
    if (PImpl->isRawPointerAbiType(ArgTy))
      reportFatalError("createCall: raw-pointer ABI calls are unsupported in "
                       "MLIR backend (memref-only). Change extern to take "
                       "memref or add LLVM-dialect lowering.");
  }

  const bool InDeviceFunc =
      PImpl->CurrentFuncOp && isa<gpu::GPUFuncOp>(PImpl->CurrentFuncOp);
  const bool InHostFunc =
      PImpl->CurrentFuncOp && isa<func::FuncOp>(PImpl->CurrentFuncOp);

  if (!InDeviceFunc && !InHostFunc)
    reportFatalError("createCall: no active function context");

  auto MapIRType = [&](IRType Ty) -> mlir::Type {
    return InDeviceFunc ? toDeviceMLIRType(Ty, PImpl->Context)
                        : toMLIRType(Ty, PImpl->Context);
  };

  llvm::SmallVector<mlir::Type> MLIRArgTys;
  MLIRArgTys.reserve(ArgTys.size());
  for (const auto &ArgTy : ArgTys)
    MLIRArgTys.push_back(MapIRType(ArgTy));

  llvm::SmallVector<mlir::Type> ResultTys;
  if (RetTy.Kind != IRTypeKind::Void)
    ResultTys.push_back(MapIRType(RetTy));

  auto FTy = PImpl->Builder.getFunctionType(MLIRArgTys, ResultTys);

  llvm::SmallVector<mlir::Value> CallArgs;
  CallArgs.reserve(Args.size());
  for (size_t I = 0; I < Args.size(); ++I) {
    mlir::Value V = PImpl->unwrap(Args[I]);
    if (PImpl->isPointerValue(V))
      reportFatalError("createCall: passing internal pointer value is "
                       "unsupported; pass a memref base instead");
    CallArgs.push_back(V);
  }

  auto Loc = PImpl->Builder.getUnknownLoc();
  func::CallOp Call;
  if (InDeviceFunc) {
    auto Callee = PImpl->lookupDeviceFunc(FName);
    if (!Callee)
      reportFatalError("createCall: unresolved device callee " + FName +
                       " (define gpu.func before calling it)");
    if (Callee.getFunctionType() != FTy)
      reportFatalError("createCall: device callee type mismatch for " + FName);

    // Device-side helper calls inside gpu.module are represented with
    // func.call symbol references to sibling gpu.func symbols.
    Call = PImpl->Builder.create<func::CallOp>(Loc, FName, FTy.getResults(),
                                               ValueRange{CallArgs});
  } else {
    auto Callee = PImpl->getOrCreateFunc(FName, FTy);
    (void)Callee;
    Call = PImpl->Builder.create<func::CallOp>(Loc, FName, FTy.getResults(),
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
    return PImpl->wrap(PImpl->Builder.create<math::SinOp>(Loc, UnwrapArg(0)));
  }
  if (Name == "cosf") {
    RequireArgCount(1);
    RequireFloatRetTy();
    return PImpl->wrap(PImpl->Builder.create<math::CosOp>(Loc, UnwrapArg(0)));
  }
  if (Name == "expf") {
    RequireArgCount(1);
    RequireFloatRetTy();
    return PImpl->wrap(PImpl->Builder.create<math::ExpOp>(Loc, UnwrapArg(0)));
  }
  if (Name == "logf") {
    RequireArgCount(1);
    RequireFloatRetTy();
    return PImpl->wrap(PImpl->Builder.create<math::LogOp>(Loc, UnwrapArg(0)));
  }
  if (Name == "sqrtf") {
    RequireArgCount(1);
    RequireFloatRetTy();
    return PImpl->wrap(PImpl->Builder.create<math::SqrtOp>(Loc, UnwrapArg(0)));
  }
  if (Name == "truncf") {
    RequireArgCount(1);
    RequireFloatRetTy();
    return PImpl->wrap(PImpl->Builder.create<math::TruncOp>(Loc, UnwrapArg(0)));
  }
  if (Name == "powf") {
    RequireArgCount(2);
    RequireFloatRetTy();
    return PImpl->wrap(
        PImpl->Builder.create<math::PowFOp>(Loc, UnwrapArg(0), UnwrapArg(1)));
  }
  if (Name == "fabsf" || Name == "absf") {
    RequireArgCount(1);
    RequireFloatRetTy();
    auto Arg = UnwrapArg(0);
    auto Neg = PImpl->Builder.create<arith::NegFOp>(Loc, Arg);
    return PImpl->wrap(PImpl->Builder.create<arith::MaximumFOp>(Loc, Arg, Neg));
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
                     "kernel");
  if (!PImpl->CurrentFuncOp || !isa<gpu::GPUFuncOp>(PImpl->CurrentFuncOp))
    reportFatalError("MLIRCodeBuilder::emitBuiltin: builtin only valid in GPU "
                     "device function");

  auto Loc = PImpl->Builder.getUnknownLoc();
  auto EmitHipImplicitArgPtr = [&]() -> mlir::Value {
    constexpr unsigned ConstantAddressSpace = 4;
    auto PtrTy =
        mlir::LLVM::LLVMPointerType::get(&PImpl->Context, ConstantAddressSpace);
#if LLVM_VERSION_MAJOR >= 20
    auto Call = PImpl->Builder.create<mlir::LLVM::CallIntrinsicOp>(
        Loc, PtrTy, PImpl->Builder.getStringAttr("llvm.amdgcn.implicitarg.ptr"),
        ValueRange{});
#else
    auto Call = PImpl->Builder.create<mlir::LLVM::CallIntrinsicOp>(
        Loc, TypeRange{PtrTy}, "llvm.amdgcn.implicitarg.ptr", ValueRange{});
#endif
    return Call.getResult(0);
  };
  auto EmitLLVMConstI64 = [&](int64_t V) -> mlir::Value {
    auto I64Ty = mlir::IntegerType::get(&PImpl->Context, 64);
    return PImpl->Builder.create<mlir::LLVM::ConstantOp>(
        Loc, I64Ty, PImpl->Builder.getI64IntegerAttr(V));
  };
  auto EmitImplicitArgLoad = [&](mlir::Type ElemTy,
                                 int64_t Offset) -> mlir::Value {
    constexpr unsigned ConstantAddressSpace = 4;
    auto PtrTy =
        mlir::LLVM::LLVMPointerType::get(&PImpl->Context, ConstantAddressSpace);
    mlir::Value ImplicitArgPtr = EmitHipImplicitArgPtr();
    mlir::Value OffsetVal = EmitLLVMConstI64(Offset);
    auto GEP = PImpl->Builder.create<mlir::LLVM::GEPOp>(
        Loc, PtrTy, ElemTy, ImplicitArgPtr, ValueRange{OffsetVal});
    return PImpl->Builder.create<mlir::LLVM::LoadOp>(Loc, ElemTy, GEP);
  };
  auto CastI32ToRetTy = [&](mlir::Value I32V) -> IRValue * {
    if (RetTy.Kind == IRTypeKind::Int32)
      return PImpl->wrap(I32V);
    if (RetTy.Kind == IRTypeKind::Int64) {
      auto I64Ty = mlir::IntegerType::get(&PImpl->Context, 64);
      auto Cast = PImpl->Builder.create<arith::ExtUIOp>(Loc, I64Ty, I32V);
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
          PImpl->Builder.create<arith::IndexCastUIOp>(Loc, DstTy, IndexV);
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
        PImpl->Builder.create<gpu::ThreadIdOp>(Loc, gpu::Dimension::x));
  if (Name == "threadIdx.y")
    return CastIndexToRetTy(
        PImpl->Builder.create<gpu::ThreadIdOp>(Loc, gpu::Dimension::y));
  if (Name == "threadIdx.z")
    return CastIndexToRetTy(
        PImpl->Builder.create<gpu::ThreadIdOp>(Loc, gpu::Dimension::z));

  if (Name == "blockIdx.x")
    return CastIndexToRetTy(
        PImpl->Builder.create<gpu::BlockIdOp>(Loc, gpu::Dimension::x));
  if (Name == "blockIdx.y")
    return CastIndexToRetTy(
        PImpl->Builder.create<gpu::BlockIdOp>(Loc, gpu::Dimension::y));
  if (Name == "blockIdx.z")
    return CastIndexToRetTy(
        PImpl->Builder.create<gpu::BlockIdOp>(Loc, gpu::Dimension::z));

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
      mlir::Value V32 = PImpl->Builder.create<arith::ExtUIOp>(Loc, I32Ty, V16);
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
        PImpl->Builder.create<gpu::BlockDimOp>(Loc, gpu::Dimension::x));
  if (Name == "blockDim.y")
    return CastIndexToRetTy(
        PImpl->Builder.create<gpu::BlockDimOp>(Loc, gpu::Dimension::y));
  if (Name == "blockDim.z")
    return CastIndexToRetTy(
        PImpl->Builder.create<gpu::BlockDimOp>(Loc, gpu::Dimension::z));

  if (Name == "gridDim.x")
    return CastIndexToRetTy(
        PImpl->Builder.create<gpu::GridDimOp>(Loc, gpu::Dimension::x));
  if (Name == "gridDim.y")
    return CastIndexToRetTy(
        PImpl->Builder.create<gpu::GridDimOp>(Loc, gpu::Dimension::y));
  if (Name == "gridDim.z")
    return CastIndexToRetTy(
        PImpl->Builder.create<gpu::GridDimOp>(Loc, gpu::Dimension::z));

  if (Name == "syncThreads") {
    PImpl->Builder.create<gpu::BarrierOp>(Loc);
    return nullptr;
  }

  reportFatalError("MLIRCodeBuilder::emitBuiltin: unsupported builtin " + Name);
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
  mlir::Value LhsSlot = PImpl->unwrap(Slot);
  auto LhsSlotTy = dyn_cast<MemRefType>(LhsSlot.getType());
  if (!LhsSlotTy || LhsSlotTy.getRank() != 1 || LhsSlotTy.getShape()[0] != 1 ||
      !LhsSlotTy.getElementType().isIndex())
    reportFatalError("storeAddress: expected pointer slot (memref<1xindex>)");

  auto LhsIt = PImpl->PointerMap.find(LhsSlot);
  mlir::Value LhsBase = (LhsIt != PImpl->PointerMap.end())
                            ? LhsIt->second.BaseMemRef
                            : mlir::Value{};

  mlir::Value RhsIdx;
  mlir::Value RhsBase;

  mlir::Value AddrV = PImpl->unwrap(Addr);
  auto AddrMemRefTy = dyn_cast<MemRefType>(AddrV.getType());
  bool IsPointerSlotTy = AddrMemRefTy && AddrMemRefTy.getRank() == 1 &&
                         AddrMemRefTy.getShape()[0] == 1 &&
                         AddrMemRefTy.getElementType().isIndex();

  mlir::Value Zero = PImpl->Builder.create<arith::ConstantIndexOp>(Loc, 0);
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
  mlir::Value Old =
      PImpl->emitAtomicRmw(Impl::AtomicOp::Add, Base, Idx, PImpl->unwrap(Val));
  return PImpl->wrap(Old);
}

IRValue *MLIRCodeBuilder::createAtomicSub(IRValue *Addr, IRValue *Val) {
  auto [Base, Idx] = PImpl->resolveAtomicAddress(Addr);
  mlir::Value Old =
      PImpl->emitAtomicRmw(Impl::AtomicOp::Sub, Base, Idx, PImpl->unwrap(Val));
  return PImpl->wrap(Old);
}

IRValue *MLIRCodeBuilder::createAtomicMax(IRValue *Addr, IRValue *Val) {
  auto [Base, Idx] = PImpl->resolveAtomicAddress(Addr);
  mlir::Value Old =
      PImpl->emitAtomicRmw(Impl::AtomicOp::Max, Base, Idx, PImpl->unwrap(Val));
  return PImpl->wrap(Old);
}

IRValue *MLIRCodeBuilder::createAtomicMin(IRValue *Addr, IRValue *Val) {
  auto [Base, Idx] = PImpl->resolveAtomicAddress(Addr);
  mlir::Value Old =
      PImpl->emitAtomicRmw(Impl::AtomicOp::Min, Base, Idx, PImpl->unwrap(Val));
  return PImpl->wrap(Old);
}

IRValue *MLIRCodeBuilder::loadFromPointee(IRValue *Slot, IRType /*AllocTy*/,
                                          IRType /*ValueTy*/) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  mlir::Value SlotV = PImpl->unwrap(Slot);
  auto It = PImpl->PointerMap.find(SlotV);
  if (It == PImpl->PointerMap.end())
    reportFatalError("loadFromPointee: unknown pointer slot");
  mlir::Value Base = It->second.BaseMemRef;
  // Load offset from the memref<1xindex> slot.
  mlir::Value IdxZero = PImpl->Builder.create<arith::ConstantIndexOp>(Loc, 0);
  mlir::Value Offset =
      PImpl->Builder.create<memref::LoadOp>(Loc, SlotV, ValueRange{IdxZero});
  // Load element from base[offset].
  mlir::Value Result =
      PImpl->Builder.create<memref::LoadOp>(Loc, Base, ValueRange{Offset});
  return PImpl->wrap(Result);
}

void MLIRCodeBuilder::storeToPointee(IRValue *Slot, IRType /*AllocTy*/,
                                     IRValue *Val) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  mlir::Value SlotV = PImpl->unwrap(Slot);
  auto It = PImpl->PointerMap.find(SlotV);
  if (It == PImpl->PointerMap.end())
    reportFatalError("storeToPointee: unknown pointer slot");
  mlir::Value Base = It->second.BaseMemRef;
  mlir::Value IdxZero = PImpl->Builder.create<arith::ConstantIndexOp>(Loc, 0);
  mlir::Value Offset =
      PImpl->Builder.create<memref::LoadOp>(Loc, SlotV, ValueRange{IdxZero});
  mlir::Value V = PImpl->unwrap(Val);
  PImpl->Builder.create<memref::StoreOp>(Loc, V, Base, ValueRange{Offset});
}

VarAlloc MLIRCodeBuilder::allocPointer(const std::string & /*Name*/,
                                       IRType ElemTy, unsigned AddrSpace) {
  auto Loc = PImpl->Builder.getUnknownLoc();
  auto IdxMemRefTy = MemRefType::get({1}, PImpl->Builder.getIndexType());
  mlir::Value OffsetSlot;
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
  mlir::Value Alloca;
  {
    OpBuilder::InsertionGuard Guard(PImpl->Builder);
    PImpl->Builder.setInsertionPointToStart(PImpl->EntryBlock);
    Alloca = PImpl->Builder.create<memref::AllocaOp>(Loc, ArrMemRefTy);
  }
  IRType AllocTy{IRTypeKind::Array, ElemTy.Signed, NElem, ElemTy.Kind};
  return {PImpl->wrap(Alloca), ElemTy, AllocTy, 0};
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
