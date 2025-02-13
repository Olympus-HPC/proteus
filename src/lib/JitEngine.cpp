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
#include <string>

#include <llvm/CodeGen/CommandFlags.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/TargetParser/SubtargetFeature.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>

#include "proteus/CoreLLVM.hpp"
#include "proteus/Hashing.hpp"
#include "proteus/JitEngine.hpp"
#include "proteus/TimeTracing.hpp"
#include "proteus/Utils.h"

// TODO: Used in InitTargetOptionsFromCodeGenFlags. Re-think for a different
// initialization, especially using static compilation flags forwarded from
// ProteusPass.
static llvm::codegen::RegisterCodeGenFlags CFG;
namespace proteus {

#if PROTEUS_ENABLE_TIME_TRACING
TimeTracerRAII TimeTracer;
#endif

using namespace llvm;

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
  Config.ENV_PROTEUS_DUMP_LLVM_IR =
      getEnvOrDefaultBool("ENV_PROTEUS_DUMP_LLVM_IR", false);
  Config.ENV_PROTEUS_RELINK_GLOBALS_BY_COPY =
      getEnvOrDefaultBool("ENV_PROTEUS_RELINK_GLOBALS_BY_COPY", false);

#if PROTEUS_ENABLE_DEBUG
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

std::string JitEngine::mangleSuffix(HashT &HashValue) {
  return "$jit$" + HashValue.toString() + "$";
}

void JitEngine::optimizeIR(Module &M, StringRef Arch, char OptLevel,
                           unsigned CodegenOptLevel) {
  TIMESCOPE("Optimize IR");
  proteus::optimizeIR(M, Arch, OptLevel, CodegenOptLevel);
}

void JitEngine::runCleanupPassPipeline(Module &M) {
  proteus::runCleanupPassPipeline(M);
}

} // namespace proteus
