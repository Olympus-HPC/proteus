//===-- JitEngine.cpp -- Base JIT Engine implementation --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include <cstdlib>
#include <llvm/IR/DebugInfo.h>
#include <llvm/Transforms/IPO/GlobalDCE.h>
#include <llvm/Transforms/IPO/StripDeadPrototypes.h>
#include <llvm/Transforms/IPO/StripSymbols.h>
#include <memory>
#include <optional>
#include <string>

#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/SubtargetFeature.h"
#include <llvm/IR/Constants.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>

#include "JitEngine.hpp"

#include "TimeTracing.hpp"
#include "Utils.h"

// TODO: Used in InitTargetOptionsFromCodeGenFlags. Re-think for a different
// initialization, especially using static compilation flags forwarded from
// ProteusPass.
static llvm::codegen::RegisterCodeGenFlags CFG;
namespace proteus {

#if ENABLE_TIME_TRACING
TimeTracerRAII TimeTracer;
#endif

using namespace llvm;

Expected<std::unique_ptr<TargetMachine>>
JitEngine::createTargetMachine(Module &M, StringRef Arch, unsigned OptLevel) {
  Triple TT(M.getTargetTriple());
  auto CGOptLevel = CodeGenOpt::getLevel(OptLevel);
  if (CGOptLevel == std::nullopt)
    FATAL_ERROR("Invalid opt level");

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

  TargetOptions Options = codegen::InitTargetOptionsFromCodeGenFlags(TT);

  std::unique_ptr<TargetMachine> TM(T->createTargetMachine(
      M.getTargetTriple(), Arch, Features.getString(), Options, RelocModel,
      CodeModel, CGOptLevel.value()));
  if (!TM)
    return make_error<StringError>("Failed to create target machine",
                                   inconvertibleErrorCode());
  return TM;
}

void JitEngine::runOptimizationPassPipeline(Module &M, StringRef Arch,
                                            unsigned OptLevel) {
  PipelineTuningOptions PTO;

  std::optional<PGOOptions> PGOOpt;
  auto TM = createTargetMachine(M, Arch, OptLevel);
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

  ModulePassManager Passes =
      PB.buildPerModuleDefaultPipeline(OptimizationLevel::O3);
  Passes.run(M, MAM);
}

JitEngine::JitEngine() {
  Config.ENV_PROTEUS_USE_STORED_CACHE =
      getEnvOrDefaultBool("ENV_PROTEUS_USE_STORED_CACHE", true);
  Config.ENV_PROTEUS_SET_LAUNCH_BOUNDS =
      getEnvOrDefaultBool("ENV_PROTEUS_SET_LAUNCH_BOUNDS", true);
  Config.ENV_PROTEUS_SPECIALIZE_ARGS =
      getEnvOrDefaultBool("ENV_PROTEUS_SPECIALIZE_ARGS", true);
  Config.ENV_PROTEUS_SPECIALIZE_DIMS =
      getEnvOrDefaultBool("ENV_PROTEUS_SPECIALIZE_DIMS", true);
  Config.ENV_PROTEUS_USE_HIP_RTC_CODEGEN =
      getEnvOrDefaultBool("ENV_PROTEUS_USE_HIP_RTC_CODEGEN", true);
  Config.ENV_PROTEUS_DISABLE =
      getEnvOrDefaultBool("ENV_PROTEUS_DISABLE", false);

#if ENABLE_DEBUG
  Logger::logs("proteus") << "ENV_PROTEUS_USE_STORED_CACHE "
                          << Config.ENV_PROTEUS_USE_STORED_CACHE << "\n";
  Logger::logs("proteus") << "ENV_PROTEUS_SET_LAUNCH_BOUNDS "
                          << Config.ENV_PROTEUS_SET_LAUNCH_BOUNDS << "\n";
  Logger::logs("proteus") << "ENV_PROTEUS_SPECIALIZE_ARGS "
                          << Config.ENV_PROTEUS_SPECIALIZE_ARGS << "\n";
  Logger::logs("proteus") << "ENV_PROTEUS_SPECIALIZE_DIMS "
                          << Config.ENV_PROTEUS_SPECIALIZE_DIMS << "\n";
  Logger::logs("proteus") << "ENV_PROTEUS_USE_HIP_RTC_CODEGEN "
                          << Config.ENV_PROTEUS_USE_HIP_RTC_CODEGEN << "\n";
#endif
}

std::string JitEngine::mangleSuffix(uint64_t HashValue) {
  return "$jit$" + std::to_string(HashValue) + "$";
}

void JitEngine::optimizeIR(Module &M, StringRef Arch) {
  TIMESCOPE("Optimize IR");
  runOptimizationPassPipeline(M, Arch);
}

void JitEngine::runCleanupPassPipeline(Module &M) {
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

} // namespace proteus
