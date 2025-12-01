#ifndef PROTEUS_CONFIG_HPP
#define PROTEUS_CONFIG_HPP

#include <string>

#include "proteus/Error.h"
#include "proteus/Logger.hpp"
#include "llvm/ADT/StringMap.h"
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>

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

inline CodegenOption strToCG(std::string CGstr) {
  std::transform(CGstr.begin(), CGstr.end(), CGstr.begin(), ::tolower);
  if (CGstr == "rtc")
    return CodegenOption::RTC;
  if (CGstr == "serial")
    return CodegenOption::Serial;
  if (CGstr == "parallel")
    return CodegenOption::Parallel;

  PROTEUS_FATAL_ERROR("Unknown codegen option: " + CGstr);
}

inline CodegenOption getEnvOrDefaultCG(const char *VarName,
                                       CodegenOption Default) {

  const char *EnvValue = std::getenv(VarName);
  if (!EnvValue)
    return Default;

  return strToCG(EnvValue);
}

template <typename T>
T getDefaultValueFromOptional(std::optional<T> JSONValue, T Default) {
  if (JSONValue)
    return JSONValue.value();
  return Default;
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
  static const bool DefaultSpecializeDimsRange;

  static CodegenOption getCodeGen(CodegenOption ProteusCodegen);

  std::optional<const std::string> ProteusOptPipeline;
  CodegenOption ProteusCodegen;
  bool ProteusSpecializeArgs;
  bool ProteusSpecializeLaunchBounds;
  bool ProteusSpecializeDims;
  bool ProteusSpecializeDimsRange;
  char ProteusOptLevel;
  int ProteusCodeGenOptLevel;
  int TunedMaxThreads;
  int MinBlocksPerSM;

  CodeGenerationConfig(std::optional<const std::string> ProteusOptPipeline,
                       CodegenOption ProteusCodegen, bool ProteusSpecializeArgs,
                       bool ProteusSpecializeLaunchBounds,
                       bool ProteusSpecializeDims,
                       bool ProteusSpecializeDimsRange, char ProteusOptLevel,
                       int ProteusCodeGenOptLevel, int TunedMaxThreads = -1,
                       int MinBlocksPerSM = 0);

public:
  static CodeGenerationConfig createFromEnv();

  static CodeGenerationConfig
  createFromJSONEntry(const llvm::json::Object &Config);

  CodegenOption codeGenOption() const;
  bool specializeArgs() const;
  bool specializeDims() const;
  bool specializeDimsRange() const;
  bool specializeLaunchBounds() const;
  char optLevel() const;
  int codeGenOptLevel() const;
  std::optional<const std::string> optPipeline() const;

  int minBlocksPerSM(int MaxThreads) const;

  template <typename T> void dump(T &OS) const {
    if (ProteusOptPipeline)
      OS << "Pipeline:" << ProteusOptPipeline.value() << " ";

    OS << "CG:" << toString(ProteusCodegen) << " ";
    OS << "SA:" << ProteusSpecializeArgs << " ";
    OS << "LB:" << ProteusSpecializeLaunchBounds << " ";
    OS << "SD:" << ProteusSpecializeDims << " ";
    OS << "SDR:" << ProteusSpecializeDimsRange << " ";
    OS << "OL:" << ProteusOptLevel << " ";
    OS << "CGL:" << ProteusCodeGenOptLevel << " ";
    OS << "TMT:" << TunedMaxThreads << " ";
    OS << "BPSM:" << MinBlocksPerSM << " ";
  }
};

inline llvm::StringMap<const CodeGenerationConfig>
parseJSONConfig(std::optional<std::string> JSONFn) {
  llvm::StringMap<const CodeGenerationConfig> TunedConfigs;
  if (!JSONFn)
    return TunedConfigs;

  auto JSONRoot = [&JSONFn]() -> std::optional<llvm::json::Object> {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> JSONBuf =
        llvm::MemoryBuffer::getFile(JSONFn.value(), /* isText */ true,
                                    /* RequiresNullTerminator */ true);
    if (!JSONBuf)
      PROTEUS_FATAL_ERROR("Error when opening json file: " +
                          JSONBuf.getError().message() + "\n");

    llvm::json::Value JsonInfo =
        llvm::cantFail(llvm::json::parse(JSONBuf.get()->getBuffer()),
                       "Cannot convert buffer to json value");

    if (auto *Obj = JsonInfo.getAsObject())
      return *Obj;
    return std::nullopt;
  }();

  if (!JSONRoot)
    PROTEUS_FATAL_ERROR("Top-level JSON is not an object.\n");

  for (auto &KV : JSONRoot.value()) {
    auto KernelName = KV.first;
    if (const auto *Options = KV.second.getAsObject()) {
      TunedConfigs.try_emplace(
          KernelName, CodeGenerationConfig::createFromJSONEntry(*Options));
    }
  }
  return TunedConfigs;
}

class Config {
public:
  static Config &get();
  const CodeGenerationConfig GlobalCodeGenConfig;
  const llvm::StringMap<const CodeGenerationConfig> TunedConfigs;
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
  bool ProteusDebugOutput;
  std::optional<const std::string> ProteusCacheDir;

  const CodeGenerationConfig &getCGConfig(llvm::StringRef KName = "") const;

  void dump(llvm::raw_ostream &OS) const;

private:
  Config();
};
} // namespace proteus

#endif
