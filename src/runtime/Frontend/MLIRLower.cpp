#include "proteus/impl/Frontend/MLIRLower.h"

#include "proteus/Error.h"
#include "proteus/impl/CoreLLVM.h"

#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/IndexToLLVM/IndexToLLVM.h>
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
#include <mlir/Dialect/Index/IR/IndexDialect.h>
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
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
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
                                           llvm::StringRef Features,
                                           llvm::StringRef Prefix) {
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
#if LLVM_VERSION_MAJOR >= 22
  llvm::Triple TT{TargetTriple};
  const llvm::Target *T = llvm::TargetRegistry::lookupTarget(TT, Error);
#else
  const llvm::Target *T =
      llvm::TargetRegistry::lookupTarget(TargetTriple, Error);
#endif
  if (!T)
    reportFatalError(Prefix.str() + ": failed to lookup LLVM target for '" +
                     TargetTriple.str() + "': " + Error);

  llvm::TargetOptions Options;
  std::optional<llvm::Reloc::Model> RelocModel;
  std::optional<llvm::CodeModel::Model> CodeModel;

  std::unique_ptr<llvm::TargetMachine> TM(T->createTargetMachine(
#if LLVM_VERSION_MAJOR >= 22
      llvm::Triple{TargetTriple},
#else
      TargetTriple,
#endif
      CPU, Features, Options, RelocModel, CodeModel,
      /*OptLevel=*/llvm::CodeGenOptLevel::Default));
  if (!TM)
    reportFatalError(Prefix.str() +
                     ": failed to create LLVM TargetMachine for '" +
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

void registerMLIRLoweringDialects(mlir::DialectRegistry &Registry) {
#if LLVM_VERSION_MAJOR >= 22
  mlir::arith::registerConvertArithToLLVMInterface(Registry);
  mlir::cf::registerConvertControlFlowToLLVMInterface(Registry);
  mlir::registerConvertFuncToLLVMInterface(Registry);
  mlir::index::registerConvertIndexToLLVMInterface(Registry);
  mlir::registerConvertMathToLLVMInterface(Registry);
  mlir::registerConvertMemRefToLLVMInterface(Registry);
#else
  (void)Registry;
#endif
}

void loadMLIRLoweringDialects(mlir::MLIRContext &Context) {
#if LLVM_VERSION_MAJOR >= 22
  mlir::DialectRegistry Registry;
  registerMLIRLoweringDialects(Registry);
  Context.appendDialectRegistry(Registry);
#endif
  Context.loadDialect<
      mlir::func::FuncDialect, arith::ArithDialect, cf::ControlFlowDialect,
      gpu::GPUDialect, index::IndexDialect, LLVM::LLVMDialect,
#if PROTEUS_ENABLE_CUDA
      NVVM::NVVMDialect,
#endif
#if PROTEUS_ENABLE_HIP
      ROCDL::ROCDLDialect,
#endif
      math::MathDialect, memref::MemRefDialect, scf::SCFDialect>();
}

static bool isStandaloneDeviceTarget(TargetModelType TM) {
  return TM == TargetModelType::CUDA || TM == TargetModelType::HIP;
}

static bool isMixedHostDeviceTarget(TargetModelType TM) {
  return TM == TargetModelType::HOST_CUDA || TM == TargetModelType::HOST_HIP;
}

static void setModuleTargetAttrs(mlir::ModuleOp Module, OpBuilder &Builder,
                                 llvm::StringRef TargetTriple,
                                 llvm::StringRef CPU, llvm::StringRef Features,
                                 llvm::StringRef Prefix) {
  const std::string DataLayout =
      computeTargetDataLayout(TargetTriple, CPU, Features, Prefix);
  Module.getOperation()->setAttr(
      mlir::LLVM::LLVMDialect::getTargetTripleAttrName(),
      Builder.getStringAttr(TargetTriple));
  Module.getOperation()->setAttr(
      mlir::LLVM::LLVMDialect::getDataLayoutAttrName(),
      Builder.getStringAttr(DataLayout));
}

#if LLVM_VERSION_MAJOR >= 22
static void addGenericToLLVMConversion(mlir::PassManager &PM) {
  mlir::ConvertToLLVMPassOptions Opts;
  Opts.filterDialects = {"arith", "func", "index", "math", "memref"};
  // Use DataLayoutAnalysis-backed conversion so index lowering follows the
  // module data layout, matching the explicit pre-LLVM-22 pipeline.
  Opts.useDynamic = true;
  PM.addPass(mlir::createConvertToLLVMPass(Opts));
  PM.addPass(mlir::createReconcileUnrealizedCastsPass());
}
#endif

static void addHostLoweringPipeline(mlir::PassManager &PM) {
  // HOST path: input is host-side structured IR (func/scf/arith/math/
  // memref); output is host-targeted LLVM dialect before translation.
#if LLVM_VERSION_MAJOR >= 22
  PM.addPass(mlir::createSCFToControlFlowPass());
#else
  PM.addPass(mlir::createConvertSCFToCFPass());
#endif
  {
    mlir::ConvertControlFlowToLLVMPassOptions Opts;
    Opts.indexBitwidth = mlir::kDeriveIndexBitwidthFromDataLayout;
    PM.addPass(mlir::createConvertControlFlowToLLVMPass(Opts));
  }
#if LLVM_VERSION_MAJOR >= 22
  addGenericToLLVMConversion(PM);
#else
  PM.addPass(mlir::createArithToLLVMConversionPass());
  PM.addPass(mlir::createConvertMathToLLVMPass());
  PM.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  PM.addPass(mlir::createConvertFuncToLLVMPass());
  PM.addPass(mlir::createReconcileUnrealizedCastsPass());
#endif
}

#if PROTEUS_ENABLE_CUDA || PROTEUS_ENABLE_HIP
static void addCommonGpuLoweringPrelude(mlir::PassManager &PM) {
#if LLVM_VERSION_MAJOR >= 22
  PM.addPass(mlir::createSCFToControlFlowPass());
#else
  PM.addPass(mlir::createConvertSCFToCFPass());
#endif
  {
    mlir::ConvertControlFlowToLLVMPassOptions Opts;
    Opts.indexBitwidth = mlir::kDeriveIndexBitwidthFromDataLayout;
    PM.addPass(mlir::createConvertControlFlowToLLVMPass(Opts));
  }
  PM.addPass(mlir::createReconcileUnrealizedCastsPass());
}

static void addCommonGpuLLVMFinalization(mlir::PassManager &PM) {
#if LLVM_VERSION_MAJOR >= 22
  addGenericToLLVMConversion(PM);
#else
  mlir::ArithToLLVMConversionPassOptions ArithOpts;
  ArithOpts.indexBitwidth = mlir::kDeriveIndexBitwidthFromDataLayout;
  PM.addPass(mlir::createArithToLLVMConversionPass(ArithOpts));

  PM.addPass(mlir::createConvertMathToLLVMPass());

  mlir::ConvertIndexToLLVMPassOptions IndexOpts;
  IndexOpts.indexBitwidth = mlir::kDeriveIndexBitwidthFromDataLayout;
  PM.addPass(mlir::createConvertIndexToLLVMPass(IndexOpts));

  mlir::FinalizeMemRefToLLVMConversionPassOptions MemRefOpts;
  MemRefOpts.indexBitwidth = mlir::kDeriveIndexBitwidthFromDataLayout;
  PM.addPass(mlir::createFinalizeMemRefToLLVMConversionPass(MemRefOpts));

  PM.addPass(mlir::createReconcileUnrealizedCastsPass());
#endif
}
#endif

#if PROTEUS_ENABLE_CUDA
static void addCUDALoweringPipeline(mlir::PassManager &PM) {
  // CUDA path: lower device-only gpu.module/gpu.func to NVVM/LLVM dialect.
  // This pass is defined on gpu.module, so schedule it as a nested pass.
  mlir::ConvertGpuOpsToNVVMOpsOptions NVVMOpts;
  NVVMOpts.indexBitwidth = mlir::kDeriveIndexBitwidthFromDataLayout;
  // Use bare pointer call convention so kernel argument ABI matches the
  // existing Proteus CUDA dispatcher launch path (raw pointer parameters).
  NVVMOpts.useBarePtrCallConv = true;

  // NOTE: ConvertControlFlowToLLVMPass is restricted to `builtin.module`,
  // so it cannot be added to a nested `gpu.module` pass manager. Run it at
  // the top-level before GPU->NVVM conversion creates llvm.func bodies.
  addCommonGpuLoweringPrelude(PM);
  PM.nest<gpu::GPUModuleOp>().addPass(
      mlir::createConvertGpuOpsToNVVMOps(NVVMOpts));
  addCommonGpuLLVMFinalization(PM);
}
#endif

#if PROTEUS_ENABLE_HIP
static void addHIPLoweringPipeline(mlir::PassManager &PM, llvm::StringRef CPU) {
  // HIP path: input is device-only gpu.module/gpu.func + gpu.* ops;
  // output is ROCDL/LLVM dialect that can be translated to AMDGPU LLVM IR.
  // Step 1 (gpu -> rocdl/llvm): lower gpu.module/gpu.func and gpu.* ops.
  // This pass is defined on gpu.module, so schedule it as a nested pass
  // instead of directly on builtin.module.
  // Use bare pointer call convention so kernel argument ABI matches the
  // existing Proteus HIP dispatcher launch path (raw pointer parameters).
  // The ROCDL lowering pass requires a non-empty chipset name.
  const std::string HipChipset = CPU.str();

  // NOTE: ConvertControlFlowToLLVMPass is restricted to `builtin.module`,
  // so it cannot be added to a nested `gpu.module` pass manager. Run it at
  // the top-level before GPU->ROCDL conversion creates llvm.func bodies.
  addCommonGpuLoweringPrelude(PM);

#if LLVM_VERSION_MAJOR >= 22
  mlir::ConvertGpuOpsToROCDLOpsOptions ROCDLOpts;
  ROCDLOpts.chipset = HipChipset;
  ROCDLOpts.indexBitwidth = mlir::kDeriveIndexBitwidthFromDataLayout;
  ROCDLOpts.useBarePtrCallConv = true;
  PM.nest<gpu::GPUModuleOp>().addPass(
      mlir::createConvertGpuOpsToROCDLOps(ROCDLOpts));
#else
  PM.nest<gpu::GPUModuleOp>().addPass(mlir::createLowerGpuOpsToROCDLOpsPass(
      /*chipset=*/HipChipset,
      /*indexBitwidth=*/mlir::kDeriveIndexBitwidthFromDataLayout,
      /*useBarePtrCallConv=*/true));
#endif
  addCommonGpuLLVMFinalization(PM);
}
#endif

static void addTargetLoweringPipeline(mlir::PassManager &PM, TargetModelType TM,
                                      llvm::StringRef CPU,
                                      llvm::StringRef Prefix) {
  if (TM == TargetModelType::HOST) {
    addHostLoweringPipeline(PM);
  } else if (TM == TargetModelType::CUDA) {
#if PROTEUS_ENABLE_CUDA
    addCUDALoweringPipeline(PM);
#else
    reportFatalError(Prefix.str() +
                     ": CUDA target requested but Proteus was built with "
                     "PROTEUS_ENABLE_CUDA=OFF");
#endif
  } else if (TM == TargetModelType::HIP) {
#if PROTEUS_ENABLE_HIP
    addHIPLoweringPipeline(PM, CPU);
#else
    reportFatalError(Prefix.str() +
                     ": HIP target requested but Proteus was built with "
                     "PROTEUS_ENABLE_HIP=OFF; requested device arch: " +
                     CPU.str());
#endif
  } else if (isMixedHostDeviceTarget(TM)) {
    reportFatalError(Prefix.str() +
                     ": mixed host+device MLIR lowering is not yet supported");
  } else {
    reportFatalError(Prefix.str() +
                     ": lowering only supported for HOST/CUDA/HIP");
  }
}

static void registerLLVMTranslations(mlir::MLIRContext &Context,
                                     TargetModelType TM,
                                     llvm::StringRef Prefix) {
  // Register dialect translations used by translateModuleToLLVMIR.
  mlir::registerBuiltinDialectTranslation(Context);
  mlir::registerLLVMDialectTranslation(Context);
  if (isStandaloneDeviceTarget(TM))
    mlir::registerGPUDialectTranslation(Context);
  if (TM == TargetModelType::CUDA) {
#if PROTEUS_ENABLE_CUDA
    mlir::registerNVVMDialectTranslation(Context);
#else
    reportFatalError(Prefix.str() +
                     ": CUDA target requested but Proteus was built with "
                     "PROTEUS_ENABLE_CUDA=OFF");
#endif
  }
  if (TM == TargetModelType::HIP) {
#if PROTEUS_ENABLE_HIP
    mlir::registerROCDLDialectTranslation(Context);
#else
    reportFatalError(Prefix.str() +
                     ": HIP target requested but Proteus was built with "
                     "PROTEUS_ENABLE_HIP=OFF");
#endif
  }
}

static gpu::GPUModuleOp getSingleDeviceModule(mlir::ModuleOp Module,
                                              llvm::StringRef Prefix) {
  llvm::SmallVector<gpu::GPUModuleOp, 2> DeviceModules;
  for (auto DeviceModule : Module.getOps<gpu::GPUModuleOp>())
    DeviceModules.push_back(DeviceModule);

  if (DeviceModules.empty())
    reportFatalError(Prefix.str() +
                     ": expected exactly one gpu.module for device lowering, "
                     "found none");

  if (DeviceModules.size() > 1) {
    std::string Message;
    llvm::raw_string_ostream OS(Message);
    OS << Prefix << ": expected exactly one gpu.module for device lowering, "
       << "found " << DeviceModules.size() << ":";
    for (gpu::GPUModuleOp DeviceModule : DeviceModules)
      OS << " @" << DeviceModule.getName();
    reportFatalError(OS.str());
  }

  return DeviceModules.front();
}

static gpu::GPUModuleOp
getDeviceModuleForSerialization(mlir::ModuleOp Module, llvm::StringRef Prefix) {
  auto DeviceSym = getSingleDeviceModule(Module, Prefix);

  mlir::PassManager DevicePM(Module.getContext());
  DevicePM.nest<gpu::GPUModuleOp>().addPass(
      mlir::createReconcileUnrealizedCastsPass());
  if (failed(DevicePM.run(Module)))
    reportFatalError(Prefix.str() +
                     ": failed to reconcile device gpu.module after lowering");

  // Re-select after running the pass manager because MLIR passes may mutate the
  // IR; this also revalidates that device lowering still has one gpu.module.
  DeviceSym = getSingleDeviceModule(Module, Prefix);

  bool HasUnrealizedCasts = false;
  DeviceSym.walk(
      [&](mlir::UnrealizedConversionCastOp) { HasUnrealizedCasts = true; });
  if (HasUnrealizedCasts) {
    llvm::errs() << "[proteus][" << Prefix
                 << "] Lowered device gpu.module still contains "
                    "builtin.unrealized_conversion_cast:\n";
    DeviceSym.print(llvm::errs());
    llvm::errs() << "\n";
    reportFatalError(Prefix.str() +
                     ": device gpu.module still contains unrealized "
                     "conversion casts after reconciliation");
  }

  return DeviceSym;
}

static std::unique_ptr<llvm::Module> serializeDeviceModuleToLLVMIR(
    Operation &OperationToSerialize, llvm::LLVMContext &LLVMCtx,
    llvm::StringRef TargetTriple, llvm::StringRef CPU, llvm::StringRef Features,
    int OptLevel, llvm::StringRef Prefix) {
  mlir::LLVM::ModuleToObject Serializer(OperationToSerialize, TargetTriple, CPU,
                                        Features, OptLevel);
  std::optional<SmallVector<char, 0>> Serialized = Serializer.run();
  if (!Serialized)
    reportFatalError(Prefix.str() +
                     ": failed to serialize device module to LLVM bitcode");

  llvm::MemoryBufferRef Buffer(
      llvm::StringRef(Serialized->data(), Serialized->size()), "DeviceBitcode");
  auto ParsedModule = llvm::parseBitcodeFile(Buffer, LLVMCtx);
  if (!ParsedModule)
    reportFatalError(Prefix.str() +
                     ": failed to parse serialized device LLVM bitcode: " +
                     llvm::toString(ParsedModule.takeError()));
  return std::move(*ParsedModule);
}

static std::unique_ptr<llvm::Module>
translateHostModuleToLLVMIR(mlir::ModuleOp Module, llvm::LLVMContext &LLVMCtx,
                            llvm::StringRef Prefix) {
  auto Mod = mlir::translateModuleToLLVMIR(Module.getOperation(), LLVMCtx);
  if (!Mod)
    reportFatalError(Prefix.str() +
                     ": failed to translate LLVM dialect to LLVM IR module");
  return Mod;
}

static void validateDeviceLLVMModule(llvm::Module &Mod,
                                     Operation &OperationToSerialize,
                                     llvm::StringRef Prefix) {
  if (hasLLVMFunctionDefinitions(Mod))
    return;

  llvm::errs() << "[proteus][" << Prefix
               << "] Device translation produced no function definitions. "
                  "MLIR device module was:\n";
  OperationToSerialize.print(llvm::errs());
  llvm::errs() << "\n";
  reportFatalError(Prefix.str() +
                   ": device translation produced an empty LLVM IR module");
}

static void normalizeDeviceKernelSymbols(llvm::Module &Mod,
                                         llvm::StringRef Prefix) {
  // MLIR device lowering may qualify kernel function symbols (e.g. with the
  // gpu.module name). Normalize known prefixes so the rest of Proteus can
  // resolve functions by their frontend names.
  llvm::SmallVector<std::pair<llvm::Function *, std::string>> ToRename;
  llvm::StringSet<> NewNames;
  for (auto &Fn : Mod.functions()) {
    if (Fn.isDeclaration())
      continue;

    llvm::StringRef Name = Fn.getName();
    static constexpr llvm::StringLiteral KernelsPrefix = "kernels.";
    if (!Name.starts_with(KernelsPrefix))
      continue;

    llvm::StringRef Stripped = Name.drop_front(KernelsPrefix.size());
    if (Stripped.empty())
      continue;

    if (Mod.getFunction(Stripped))
      reportFatalError(Prefix.str() + ": device symbol name collision for " +
                       Stripped.str());

    if (!NewNames.insert(Stripped).second)
      reportFatalError(Prefix.str() + ": multiple device symbols map to " +
                       Stripped.str());

    ToRename.emplace_back(&Fn, Stripped.str());
  }

  for (auto &[Fn, NewName] : ToRename)
    Fn->setName(NewName);
}

static void setFinalLLVMTargetAttrs(llvm::Module &Mod, TargetModelType TM,
                                    llvm::StringRef TargetTriple,
                                    llvm::StringRef CPU,
                                    llvm::StringRef Features,
                                    llvm::StringRef Prefix) {
  if (!TargetTriple.empty()) {
#if LLVM_VERSION_MAJOR >= 22
    Mod.setTargetTriple(llvm::Triple{TargetTriple});
#else
    Mod.setTargetTriple(TargetTriple);
#endif
    if (isStandaloneDeviceTarget(TM))
      Mod.setDataLayout(
          computeTargetDataLayout(TargetTriple, CPU, Features, Prefix));
  }
}

MLIRLoweringResult lowerMLIRModuleToLLVM(mlir::ModuleOp Module,
                                         const MLIRLoweringOptions &Options) {
  const TargetModelType TM = Options.TargetModel;
  const std::string EffectiveTriple =
      Options.TargetTriple.empty() ? getTargetTriple(TM) : Options.TargetTriple;
  const llvm::StringRef TargetTriple{EffectiveTriple};
  const llvm::StringRef CPU{Options.DeviceArch};
  const llvm::StringRef Features{Options.Features};
  const llvm::StringRef Prefix{Options.DiagnosticPrefix};

  if (failed(mlir::verify(Module)))
    reportFatalError(Prefix.str() + ": invalid MLIR module before lowering");

  MLIRContext *Context = Module.getContext();
  OpBuilder Builder(Context);

  // Set triple + data layout so index bitwidth derivation is consistent.
  if ((TM == TargetModelType::HIP || TM == TargetModelType::HOST_HIP) &&
      CPU.empty())
    reportFatalError(Prefix.str() +
                     ": HIP device lowering requires a non-empty device arch "
                     "(call setDeviceArch(\"gfx*\"))");
  setModuleTargetAttrs(Module, Builder, TargetTriple, CPU, Features, Prefix);

  mlir::PassManager PM(Context);
  addTargetLoweringPipeline(PM, TM, CPU, Prefix);
  if (failed(PM.run(Module)))
    reportFatalError(Prefix.str() +
                     ": failed to lower MLIR module to LLVM dialect");

  MLIRLoweringResult Result;
  Result.Ctx = std::make_unique<llvm::LLVMContext>();
  registerLLVMTranslations(*Context, TM, Prefix);

  Operation *OperationToSerialize = Module.getOperation();
  if (isStandaloneDeviceTarget(TM)) {
    auto DeviceModule = getDeviceModuleForSerialization(Module, Prefix);
    OperationToSerialize = DeviceModule.getOperation();
    Result.Mod = serializeDeviceModuleToLLVMIR(
        *OperationToSerialize, *Result.Ctx, TargetTriple, CPU, Features,
        Options.OptLevel, Prefix);
    validateDeviceLLVMModule(*Result.Mod, *OperationToSerialize, Prefix);
    normalizeDeviceKernelSymbols(*Result.Mod, Prefix);
  } else {
    Result.Mod = translateHostModuleToLLVMIR(Module, *Result.Ctx, Prefix);
  }

  setFinalLLVMTargetAttrs(*Result.Mod, TM, TargetTriple, CPU, Features, Prefix);

  return Result;
}

} // namespace proteus
