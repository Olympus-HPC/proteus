#ifndef PROTEUS_CORE_LLVM_H
#define PROTEUS_CORE_LLVM_H

static_assert(__cplusplus >= 201703L,
              "This header requires C++17 or later due to LLVM.");

#include "proteus/Error.h"
#include "proteus/impl/Debug.h"
#include "proteus/impl/Logger.h"
#include "proteus/impl/TimeTracing.h"

#include <llvm/CodeGen/CommandFlags.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Linker/Linker.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Transforms/IPO/MergeFunctions.h>

#if LLVM_VERSION_MAJOR >= 18
#include <llvm/TargetParser/SubtargetFeature.h>
// This convoluted logic below is because AMD ROCm 5.7.1 identifies as LLVM 17
// but includes the header SubtargetFeature.h to a different directory than
// upstream LLVM 17. We basically detect if it's the HIP version and include it
// from the expected MC directory, otherwise from TargetParser.
#elif LLVM_VERSION_MAJOR == 17
#if defined(__HIP_PLATFORM_HCC__) || defined(HIP_VERSION_MAJOR)
#include <llvm/MC/SubtargetFeature.h>
#else
#include <llvm/TargetParser/SubtargetFeature.h>
#endif
#else
#define STRINGIFY_HELPER(x) #x
#define STRINGIFY(x) STRINGIFY_HELPER(x)
#error "Unsupported LLVM version " STRINGIFY(LLVM_VERSION_MAJOR)
#endif
#include <llvm/Transforms/IPO/GlobalDCE.h>
#include <llvm/Transforms/IPO/Internalize.h>
#include <llvm/Transforms/IPO/StripDeadPrototypes.h>
#include <llvm/Transforms/IPO/StripSymbols.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>

namespace proteus {
using namespace llvm;

namespace detail {

inline Expected<std::unique_ptr<TargetMachine>>
createTargetMachine(Module &M, StringRef Arch, unsigned OptLevel = 3) {
  Triple TT(M.getTargetTriple());
  auto CGOptLevel = CodeGenOpt::getLevel(OptLevel);
  if (CGOptLevel == std::nullopt)
    reportFatalError("Invalid opt level");

  std::string Msg;
  const Target *T = TargetRegistry::lookupTarget(M.getTargetTriple(), Msg);
  if (!T)
    return make_error<StringError>(Msg, inconvertibleErrorCode());

  SubtargetFeatures Features;
  Features.getDefaultSubtargetFeatures(TT);

  std::optional<Reloc::Model> RelocModel;
  if (M.getModuleFlag("PIC Level"))
    RelocModel =
        M.getPICLevel() == PICLevel::NotPIC ? Reloc::Static : Reloc::PIC_;

  std::optional<CodeModel::Model> CodeModel = M.getCodeModel();

  // Use default target options.
  // TODO: Customize based on AOT compilation flags or by creating a
  // constructor that sets target options based on the triple.
  TargetOptions Options;
  std::unique_ptr<TargetMachine> TM(T->createTargetMachine(
      M.getTargetTriple(), Arch, Features.getString(), Options, RelocModel,
      CodeModel, CGOptLevel.value()));
  if (!TM)
    return make_error<StringError>("Failed to create target machine",
                                   inconvertibleErrorCode());
  return TM;
}

inline void runOptimizationPassPipeline(Module &M, StringRef Arch,
                                        const std::string &PassPipeline,
                                        unsigned CodegenOptLevel = 3) {
  PipelineTuningOptions PTO;

  std::optional<PGOOptions> PGOOpt;
  auto TM = createTargetMachine(M, Arch, CodegenOptLevel);
  if (auto Err = TM.takeError())
    report_fatal_error(std::move(Err));
  TargetLibraryInfoImpl TLII(Triple(M.getTargetTriple()));

  PassBuilder PB(TM->get(), PTO, PGOOpt, nullptr);
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  FAM.registerPass([&] { return TargetLibraryAnalysis(TLII); });

  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
  ModulePassManager Passes;
  if (auto E = PB.parsePassPipeline(Passes, PassPipeline))
    reportFatalError("Error: " + toString(std::move(E)));

  Passes.run(M, MAM);
}

inline void runOptimizationPassPipeline(Module &M, StringRef Arch,
                                        char OptLevel = '3',
                                        unsigned CodegenOptLevel = 3) {
  PipelineTuningOptions PTO;

  std::optional<PGOOptions> PGOOpt;
  auto TM = createTargetMachine(M, Arch, CodegenOptLevel);
  if (auto Err = TM.takeError())
    report_fatal_error(std::move(Err));
  TargetLibraryInfoImpl TLII(Triple(M.getTargetTriple()));

  PassBuilder PB(TM->get(), PTO, PGOOpt, nullptr);
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  FAM.registerPass([&] { return TargetLibraryAnalysis(TLII); });

  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  OptimizationLevel OptSetting;
  switch (OptLevel) {
  case '0':
    OptSetting = OptimizationLevel::O0;
    break;
  case '1':
    OptSetting = OptimizationLevel::O1;
    break;
  case '2':
    OptSetting = OptimizationLevel::O2;
    break;
  case '3':
    OptSetting = OptimizationLevel::O3;
    break;
  case 's':
    OptSetting = OptimizationLevel::Os;
    break;
  case 'z':
    OptSetting = OptimizationLevel::Oz;
    break;
  default:
    reportFatalError(std::string("Unsupported optimization level ") + OptLevel);
  };

  ModulePassManager Passes = PB.buildPerModuleDefaultPipeline(OptSetting);
  Passes.run(M, MAM);
}

} // namespace detail

struct InitLLVMTargets {
  InitLLVMTargets() {
    InitializeAllTargetInfos();
    InitializeAllTargets();
    InitializeAllTargetMCs();
    InitializeAllAsmParsers();
    InitializeAllAsmPrinters();
  }
};

inline void optimizeIR(Module &M, StringRef Arch, char OptLevel,
                       unsigned CodegenOptLevel) {
  Timer T;
  detail::runOptimizationPassPipeline(M, Arch, OptLevel, CodegenOptLevel);
  PROTEUS_TIMER_OUTPUT(Logger::outs("proteus")
                       << "optimizeIR optlevel " << OptLevel << " codegenopt "
                       << CodegenOptLevel << " " << T.elapsed() << " ms\n");
}

inline void optimizeIR(Module &M, StringRef Arch,
                       const std::string &PassPipeline,
                       unsigned CodegenOptLevel) {
  Timer T;
  auto TraceOut = [](const std::string &PassPipeline) {
    SmallString<128> S;
    raw_svector_ostream OS(S);
    OS << "[CustomPipeline] " << PassPipeline << "\n";
    return S;
  };

  if (Config::get().traceSpecializations())
    Logger::trace(TraceOut(PassPipeline));

  detail::runOptimizationPassPipeline(M, Arch, PassPipeline, CodegenOptLevel);
  PROTEUS_TIMER_OUTPUT(Logger::outs("proteus")
                       << "optimizeIR optlevel " << PassPipeline
                       << " codegenopt " << CodegenOptLevel << " "
                       << T.elapsed() << " ms\n");
}

inline std::unique_ptr<Module>
linkModules(LLVMContext &Ctx,
            SmallVector<std::unique_ptr<Module>> LinkedModules) {
  if (LinkedModules.empty())
    reportFatalError("Expected jit module");

  auto LinkedModule = std::make_unique<llvm::Module>("JitModule", Ctx);
  Linker IRLinker(*LinkedModule);
  // Link in all the proteus-enabled extracted modules.
  for (auto &LinkedM : LinkedModules) {
    // Returns true if linking failed.
    if (IRLinker.linkInModule(std::move(LinkedM)))
      reportFatalError("Linking failed");
  }

  return LinkedModule;
}

inline void runCleanupPassPipeline(Module &M) {
  PassBuilder PB;
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  ModulePassManager Passes;
  Passes.addPass(GlobalDCEPass());
  // Passes.addPass(StripDeadDebugInfoPass());
  Passes.addPass(StripDeadPrototypesPass());

  Passes.run(M, MAM);

  StripDebugInfo(M);
}

inline void pruneIR(Module &M, bool UnsetExternallyInitialized = true) {
  // Remove llvm.global.annotations now that we have read them.
  if (auto *GlobalAnnotations = M.getGlobalVariable("llvm.global.annotations"))
    M.eraseGlobalVariable(GlobalAnnotations);

  // Remove llvm.compiler.used
  if (auto *CompilerUsed = M.getGlobalVariable("llvm.compiler.used"))
    M.eraseGlobalVariable(CompilerUsed);

  // Remove the __clang_gpu_used_external used in HIP RDC compilation and its
  // uses in llvm.used, llvm.compiler.used.
  SmallVector<GlobalVariable *> GlobalsToErase;
  for (auto &GV : M.globals()) {
    auto Name = GV.getName();
    if (Name.starts_with("__clang_gpu_used_external") ||
        Name.starts_with("_jit_bitcode") || Name.starts_with("__hip_cuid")) {
      GlobalsToErase.push_back(&GV);
      removeFromUsedLists(M, [&GV](Constant *C) {
        if (auto *Global = dyn_cast<GlobalVariable>(C))
          return Global == &GV;
        return false;
      });
    }
  }
  for (auto *GV : GlobalsToErase) {
    M.eraseGlobalVariable(GV);
  }

  // Remove externaly_initialized attributes.
  if (UnsetExternallyInitialized)
    for (auto &GV : M.globals())
      if (GV.isExternallyInitialized())
        GV.setExternallyInitialized(false);
}

inline void internalize(Module &M, StringRef PreserveFunctionName) {
  auto *F = M.getFunction(PreserveFunctionName);
  // Internalize others besides the kernel function.
  internalizeModule(M, [&F](const GlobalValue &GV) {
    // Do not internalize the kernel function.
    if (&GV == F)
      return true;

    // Internalize everything else.
    return false;
  });
}

} // namespace proteus

#endif
