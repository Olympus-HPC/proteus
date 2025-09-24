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

inline char getEnvOrDefaultChar(const char *VarName, char Default) {

  const char *EnvValue = std::getenv(VarName);
  return EnvValue ? EnvValue[0] : Default;
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

class CodeGenerationConfig {
  static CodegenOption getCodeGen(CodegenOption ProteusCodegen) {
    constexpr bool SupportOnlyRTC =
#if defined(PROTEUS_ENABLE_CUDA)
        true;
#elif defined(PROTEUS_ENABLE_HIP) && (LLVM_VERSION_MAJOR < 18)
        true;
#else
        false;
#endif
    if (SupportOnlyRTC && ProteusCodegen != CodegenOption::RTC) {
      Logger::outs("proteus") << "Warning: Proteus supports only RTC in the "
                                 "current build system configuration, "
                                 "defaulting Codegen to RTC\n";
      ProteusCodegen = CodegenOption::RTC;
    }
    return ProteusCodegen;
  }

  std::optional<const std::string> ProteusOptPipeline;
  CodegenOption ProteusCodegen;
  bool ProteusSpecializeArgs;
  bool ProteusSpecializeLaunchBounds;
  bool ProteusSpecializeDims;
  bool ProteusSpecializeDimsAssume;
  char ProteusOptLevel;
  int ProteusCodeGenOptLevel;

  CodeGenerationConfig(std::optional<const std::string> ProteusOptPipeline,
                       CodegenOption ProteusCodegen, bool ProteusSpecializeArgs,
                       bool ProteusSpecializeLaunchBounds,
                       bool ProteusSpecializeDims,
                       bool ProteusSpecializeDimsAssume, char ProteusOptLevel,
                       int ProteusCodeGenOptLevel)
      : ProteusOptPipeline(ProteusOptPipeline), ProteusCodegen(ProteusCodegen),
        ProteusSpecializeArgs(ProteusSpecializeArgs),
        ProteusSpecializeLaunchBounds(ProteusSpecializeLaunchBounds),
        ProteusSpecializeDims(ProteusSpecializeDims),
        ProteusSpecializeDimsAssume(ProteusSpecializeDimsAssume),
        ProteusOptLevel(ProteusOptLevel),
        ProteusCodeGenOptLevel(ProteusCodeGenOptLevel) {}

public:
  static CodeGenerationConfig createFromEnv() {
    constexpr bool DefaultSpecializeDimsAssume =
#if PROTEUS_ENABLE_CUDA
        false;
#else
        true;
#endif

    return CodeGenerationConfig(
        getEnvOrDefaultString("PROTEUS_OPT_PIPELINE"),
        getCodeGen(getEnvOrDefaultCG("PROTEUS_CODEGEN", CodegenOption::RTC)),
        getEnvOrDefaultBool("PROTEUS_SPECIALIZE_ARGS", true),
        getEnvOrDefaultBool("PROTEUS_SPECIALIZE_LAUNCH_BOUNDS", true),
        getEnvOrDefaultBool("PROTEUS_SPECIALIZE_DIMS", true),
        getEnvOrDefaultBool("PROTEUS_SPECIALIZE_DIMS_ASSUME",
                            DefaultSpecializeDimsAssume),
        getEnvOrDefaultChar("PROTEUS_OPT_LEVEL", '3'),
        getEnvOrDefaultInt("PROTEUS_CODE_GEN_OPT_LEVEL", 3));
  }

  CodegenOption codeGenOption() const { return ProteusCodegen; }
  bool specializeArgs() const { return ProteusSpecializeArgs; }
  bool specializeDims() const { return ProteusSpecializeDims; }
  bool specializeDimsAssume() const { return ProteusSpecializeDimsAssume; }
  bool specializeLaunchBounds() const { return ProteusSpecializeLaunchBounds; }
  char optLevel() const { return ProteusOptLevel; }
  int codeGenOptLevel() const { return ProteusCodeGenOptLevel; }
  std::optional<const std::string> optPipeline() const {
    return ProteusOptPipeline;
  }
};

class Config {
public:
  static Config &get() {
    static Config Conf;
    return Conf;
  }
  const CodeGenerationConfig CodeGenConfig;
  bool ProteusUseStoredCache;
  bool ProteusDisable;
  bool ProteusDumpLLVMIR;
  bool ProteusRelinkGlobalsByCopy;
  bool ProteusAsyncCompilation;
  int ProteusAsyncThreads;
  bool ProteusAsyncTestBlocking;
  KernelCloneOption ProteusKernelClone;
  bool ProteusEnableTimers;
  int ProteusTraceOutput;
  std::optional<const std::string> ProteusCacheDir;

  const CodeGenerationConfig &getCGConfig() const { return CodeGenConfig; }

private:
  Config()
      : CodeGenConfig(CodeGenerationConfig::createFromEnv()),
        ProteusCacheDir(getEnvOrDefaultString("PROTEUS_CACHE_DIR")) {
    ProteusUseStoredCache =
        getEnvOrDefaultBool("PROTEUS_USE_STORED_CACHE", true);
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
