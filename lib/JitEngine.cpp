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
#include <memory>
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

#include "Utils.h"

namespace proteus {

using namespace llvm;

Expected<std::unique_ptr<TargetMachine>>
JitEngine::createTargetMachine(Module &M,
                               StringRef CPU /*, unsigned OptLevel*/) {
  Triple TT(M.getTargetTriple());
  // TODO: Parameterize optlevel.
  CodeGenOpt::Level CGOptLevel = CodeGenOpt::Aggressive;

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

  std::unique_ptr<TargetMachine> TM(
      T->createTargetMachine(M.getTargetTriple(), CPU, Features.getString(),
                             Options, RelocModel, CodeModel, CGOptLevel));
  if (!TM)
    return make_error<StringError>("Failed to create target machine",
                                   inconvertibleErrorCode());
  return TM;
}

void JitEngine::runOptimizationPassPipeline(Module &M, StringRef CPU) {
  TIMESCOPE("Run opt passes");
  PipelineTuningOptions PTO;

  std::optional<PGOOptions> PGOOpt;
  auto TM = createTargetMachine(M, CPU);
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
  Config.ENV_JIT_USE_STORED_CACHE =
      getEnvOrDefaultBool("ENV_JIT_USE_STORED_CACHE", true);
  Config.ENV_JIT_LAUNCH_BOUNDS =
      getEnvOrDefaultBool("ENV_JIT_LAUNCH_BOUNDS", true);
  Config.ENV_JIT_SPECIALIZE_ARGS =
      getEnvOrDefaultBool("ENV_JIT_SPECIALIZE_ARGS", true);

#if ENABLE_DEBUG
  dbgs() << "ENV_JIT_USE_STORED_CACHE " << Config.ENV_JIT_USE_STORED_CACHE
         << "\n";
  dbgs() << "ENV_JIT_LAUNCH_BOUNDS " << Config.ENV_JIT_LAUNCH_BOUNDS << "\n";
  dbgs() << "ENV_JIT_SPECIALIZE_ARGS " << Config.ENV_JIT_SPECIALIZE_ARGS
         << "\n";
#endif
}

bool JitEngine::getEnvOrDefaultBool(const char *VarName, bool Default) {
  const char *EnvValue = std::getenv(VarName);
  return EnvValue ? static_cast<bool>(std::stoi(EnvValue)) : Default;
}

std::string JitEngine::mangleSuffix(uint64_t HashValue) {
  return "$jit$" + std::to_string(HashValue) + "$";
}

} // namespace proteus
