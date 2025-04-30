#ifndef PROTEUS_CORE_LLVM_HPP
#define PROTEUS_CORE_LLVM_HPP

static_assert(__cplusplus >= 201703L,
              "This header requires C++17 or later due to LLVM.");

#include <llvm/CodeGen/CommandFlags.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Linker/Linker.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>

#if LLVM_VERSION_MAJOR == 18 or 1
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
#error "Unsupported LLVM version " STRINGIFY(LLVM_VERSION)
#endif
#include <llvm/Transforms/IPO/GlobalDCE.h>
#include <llvm/Transforms/IPO/Internalize.h>
#include <llvm/Transforms/IPO/StripDeadPrototypes.h>
#include <llvm/Transforms/IPO/StripSymbols.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>
//#include "polly/RegisterPasses.h"
#include "polly/ScopInfo.h"
#include "polly/ScopPass.h"
#include "polly/DependenceInfo.h"
#include "polly/LinkAllPasses.h"
#include "polly/RegisterPasses.h"
#include "polly/Canonicalization.h"
#include "polly/CodeGen/CodeGeneration.h"
#include "polly/CodeGen/IslAst.h"
#include "polly/CodePreparation.h"
#include "polly/DeLICM.h"
#include "polly/DeadCodeElimination.h"
#include "polly/DependenceInfo.h"
#include "polly/ForwardOpTree.h"
#include "polly/JSONExporter.h"
#include "polly/LinkAllPasses.h"
#include "polly/MaximalStaticExpansion.h"
#include "polly/PolyhedralInfo.h"
#include "polly/PruneUnprofitable.h"
#include "polly/ScheduleOptimizer.h"
#include "polly/ScopDetection.h"
#include "polly/ScopGraphPrinter.h"
#include "polly/ScopInfo.h"
#include "polly/Simplify.h"
#include "polly/Support/DumpFunctionPass.h"
#include "polly/Support/DumpModulePass.h"

#include "proteus/Error.h"

namespace proteus {
using namespace llvm;

namespace detail {

inline Expected<std::unique_ptr<TargetMachine>>
createTargetMachine(Module &M, StringRef Arch, unsigned OptLevel = 3) {
  Triple TT(M.getTargetTriple());
  auto CGOptLevel = CodeGenOpt::getLevel(OptLevel);
  if (CGOptLevel == std::nullopt)
    PROTEUS_FATAL_ERROR("Invalid opt level");

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

struct ScopPrinterPass : public llvm::PassInfoMixin<ScopPrinterPass> {
  llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM) {
      // Get ScopInfo for the function
      auto &SD = FAM.getResult<polly::ScopAnalysis>(F);

      // Print detected SCoPs
      if (SD.ValidRegions.size()) {
          llvm::outs() << "Detected SCoPs in function: " << F.getName() << "\n";
          for (auto* Region : SD.ValidRegions) {
            Region->print(llvm::outs());
          }
      } else {
          llvm::outs() << "No SCoPs found in function: " << F.getName() << "\n";
      }

      return llvm::PreservedAnalyses::all();
  }
};

inline void runOptimizationPassPipeline(Module &M, StringRef Arch,
                                        char OptLevel = '3',
                                        unsigned CodegenOptLevel = 3, bool UsePolly=true) {
  PipelineTuningOptions PTO;

  std::optional<PGOOptions> PGOOpt;
  auto TM = createTargetMachine(M, Arch, CodegenOptLevel);
  if (auto Err = TM.takeError())
    report_fatal_error(std::move(Err));
  TargetLibraryInfoImpl TLII(Triple(M.getTargetTriple()));
using namespace polly;
  PassBuilder PB(TM->get(), PTO, PGOOpt, nullptr);
  ScopPassManager SPM;
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  FunctionPassManager PM;
  ModuleAnalysisManager MAM;

  if (UsePolly) {
    //polly::initializePolly(PB);
    polly::registerPollyPasses(PB);
    PM.addPass(CodePreparationPass());

    PM.addPass(ScopViewer());
    PM.addPass(ScopOnlyViewer());
    PM.addPass(ScopPrinter());
    PM.addPass(ScopOnlyPrinter());
    SPM.addPass(SimplifyPass(0));
    SPM.addPass(ForwardOpTreePass());
    SPM.addPass(DeLICMPass());
    SPM.addPass(SimplifyPass(1));
    SPM.addPass(DeadCodeElimPass());
    SPM.addPass(MaximalStaticExpansionPass());
    SPM.addPass(IslScheduleOptimizerPass());
    SPM.addPass(CodeGenerationPass());
    
    PM.addPass(createFunctionToScopPassAdaptor(std::move(SPM)));
    PM.addPass(PB.buildFunctionSimplificationPipeline(
      OptimizationLevel::O3, llvm::ThinOrFullLTOPhase::None)); // Cleanup
  }
  //FPM.addPass(polly::createScopInfoRegionPassPass());

  // Add Custom Scop Printer Pass
  //FPM.addPass(ScopPrinterPass());

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
    PROTEUS_FATAL_ERROR("Unsupported optimization level " + OptLevel);
  };

  ModulePassManager Passes = PB.buildPerModuleDefaultPipeline(OptSetting);

  llvm::DebugFlag = true;
  llvm::outs() << "test\n";
  if (UsePolly) {
    llvm::outs() << "polly\n";
  }
  Passes.addPass(createModuleToFunctionPassAdaptor(std::move(PM)));
  Passes.printPipeline(llvm::outs(), [&](llvm::StringRef) {return llvm::StringRef{}; });
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
                       unsigned CodegenOptLevel,
                       bool use_polly=true) {
  detail::runOptimizationPassPipeline(M, Arch, OptLevel, CodegenOptLevel, use_polly);
}

inline std::unique_ptr<Module>
linkModules(LLVMContext &Ctx,
            SmallVector<std::unique_ptr<Module>> &LinkedModules) {
  if (LinkedModules.empty())
    PROTEUS_FATAL_ERROR("Expected jit module");

  auto LinkedModule = std::make_unique<llvm::Module>("JitModule", Ctx);
  Linker IRLinker(*LinkedModule);
  // Link in all the proteus-enabled extracted modules.
  for (auto &LinkedM : LinkedModules) {
    // Returns true if linking failed.
    if (IRLinker.linkInModule(std::move(LinkedM)))
      PROTEUS_FATAL_ERROR("Linking failed");
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

#include "proteus/CoreLLVMDevice.hpp"

#endif
