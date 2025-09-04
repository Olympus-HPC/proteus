#ifndef PROTEUS_CONFIG_HPP
#define PROTEUS_CONFIG_HPP

#include <string>

#include "proteus/Error.h"
#include "proteus/Logger.hpp"

namespace proteus {

enum class CodegenOption {
  RTC,
  Serial,
  Parallel,
};

enum class KernelCloneOption {
  LinkClonePrune,
  LinkCloneLight,
  CrossClone,
};

inline std::string toString(CodegenOption Option) {
  switch (Option) {
  case CodegenOption::RTC:
    return "RTC";
  case CodegenOption::Serial:
    return "Serial";
  case CodegenOption::Parallel:
    return "Parallel";
  default:
    return "Unknown";
  }
}

inline std::string toString(KernelCloneOption Option) {
  switch (Option) {
  case KernelCloneOption::LinkClonePrune:
    return "link-clone-prune";
  case proteus::KernelCloneOption::LinkCloneLight:
    return "link-clone-light";
  case proteus::KernelCloneOption::CrossClone:
    return "cross-clone";
  default:
    return "Unknown";
  }
}

inline std::optional<std::string> getEnvOrDefaultString(const char *VarName) {

  const char *EnvValue = std::getenv(VarName);
  if (!EnvValue)
    return std::nullopt;

  return std::string(EnvValue);
}

inline bool getEnvOrDefaultBool(const char *VarName, bool Default) {

  const char *EnvValue = std::getenv(VarName);
  return EnvValue ? static_cast<bool>(std::stoi(EnvValue)) : Default;
}

inline int getEnvOrDefaultInt(const char *VarName, int Default) {

  const char *EnvValue = std::getenv(VarName);
  return EnvValue ? std::stoi(EnvValue) : Default;
}

inline CodegenOption getEnvOrDefaultCG(const char *VarName,
                                       CodegenOption Default) {

  const char *EnvValue = std::getenv(VarName);
  if (!EnvValue)
    return Default;

  std::string EnvValueStr{EnvValue};
  std::transform(EnvValueStr.begin(), EnvValueStr.end(), EnvValueStr.begin(),
                 ::tolower);
  if (EnvValueStr == "rtc")
    return CodegenOption::RTC;
  if (EnvValueStr == "serial")
    return CodegenOption::Serial;
  if (EnvValueStr == "parallel")
    return CodegenOption::Parallel;

  PROTEUS_FATAL_ERROR("Unknown codegen option: " + EnvValueStr);
}

inline KernelCloneOption getEnvOrDefaultKC(const char *VarName,
                                           KernelCloneOption Default) {

  const char *EnvValue = std::getenv(VarName);
  if (!EnvValue)
    return Default;

  std::string EnvValueStr{EnvValue};
  std::transform(EnvValueStr.begin(), EnvValueStr.end(), EnvValueStr.begin(),
                 ::tolower);
  if (EnvValueStr == "link-clone-prune")
    return KernelCloneOption::LinkClonePrune;
  if (EnvValueStr == "link-clone-light")
    return KernelCloneOption::LinkCloneLight;
  if (EnvValueStr == "cross-clone")
    return KernelCloneOption::CrossClone;

  Logger::outs("proteus") << "Unknown kernel clone option " << EnvValueStr
                          << ", using default: " << toString(Default) << "\n";
  return Default;
}

class Config {
public:
  static Config &get() {
    static Config Conf;
    return Conf;
  }

  bool ProteusUseStoredCache;
  bool ProteusSpecializeLaunchBounds;
  bool ProteusSpecializeArgs;
  bool ProteusSpecializeDims;
  bool ProteusDisable;
  bool ProteusDumpLLVMIR;
  bool ProteusRelinkGlobalsByCopy;
  bool ProteusAsyncCompilation;
  int ProteusAsyncThreads;
  bool ProteusAsyncTestBlocking;
  KernelCloneOption ProteusKernelClone;
  bool ProteusEnableTimers;
  CodegenOption ProteusCodegen;
  int ProteusTraceOutput;
  std::optional<const std::string> ProteusOptPipeline;

private:
  Config() : ProteusOptPipeline(getEnvOrDefaultString("PROTEUS_OPT_PIPELINE")) {
    ProteusUseStoredCache =
        getEnvOrDefaultBool("PROTEUS_USE_STORED_CACHE", true);
    ProteusSpecializeLaunchBounds =
        getEnvOrDefaultBool("PROTEUS_SPECIALIZE_LAUNCH_BOUNDS", true);
    ProteusSpecializeArgs =
        getEnvOrDefaultBool("PROTEUS_SPECIALIZE_ARGS", true);
    ProteusSpecializeDims =
        getEnvOrDefaultBool("PROTEUS_SPECIALIZE_DIMS", true);
    ProteusCodegen = getEnvOrDefaultCG("PROTEUS_CODEGEN", CodegenOption::RTC);
#if PROTEUS_ENABLE_CUDA
    if (ProteusCodegen != CodegenOption::RTC) {
      Logger::outs("proteus")
          << "Warning: Proteus supports only RTC compilation for CUDA, "
             "setting Codegen to RTC\n";
      ProteusCodegen = CodegenOption::RTC;
    }
#endif
#if PROTEUS_ENABLE_HIP
#if LLVM_VERSION_MAJOR < 18
    if (ProteusCodegen != CodegenOption::RTC) {
      Logger::outs("proteus")
          << "Warning: Proteus with LLVM < 18 supports only RTC compilation, "
             "setting Codegen to RTC\n";
      ProteusCodegen = CodegenOption::RTC;
    }
#endif
#endif
    ProteusDisable = getEnvOrDefaultBool("PROTEUS_DISABLE", false);
    ProteusDumpLLVMIR = getEnvOrDefaultBool("PROTEUS_DUMP_LLVM_IR", false);
    ProteusRelinkGlobalsByCopy =
        getEnvOrDefaultBool("PROTEUS_RELINK_GLOBALS_BY_COPY", false);
    ProteusAsyncCompilation =
        getEnvOrDefaultBool("PROTEUS_ASYNC_COMPILATION", false);
    ProteusAsyncTestBlocking =
        getEnvOrDefaultBool("PROTEUS_ASYNC_TEST_BLOCKING", false);
    ProteusAsyncThreads = getEnvOrDefaultInt("PROTEUS_ASYNC_THREADS", 1);
    ProteusKernelClone = getEnvOrDefaultKC("PROTEUS_KERNEL_CLONE",
                                           KernelCloneOption::CrossClone);
    ProteusEnableTimers = getEnvOrDefaultBool("PROTEUS_ENABLE_TIMERS", false);
    ProteusTraceOutput = getEnvOrDefaultInt("PROTEUS_TRACE_OUTPUT", 0);
  }
};
} // namespace proteus

#endif
