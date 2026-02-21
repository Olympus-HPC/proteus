#ifndef PROTEUS_CONFIG_H
#define PROTEUS_CONFIG_H

#include "proteus/Error.h"
#include "proteus/impl/Logger.h"

#include "llvm/ADT/StringMap.h"
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>

#include <algorithm>
#include <sstream>
#include <string>
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

  reportFatalError("Unknown codegen option: " + CGstr);
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

enum class TraceOption : unsigned {
  Specialization = 0x1,
  IRDump = 0x2,
  KernelTrace = 0x4,
};

inline unsigned parseTraceConfig(const char *VarName) {
  const char *EnvValue = std::getenv(VarName);
  if (!EnvValue)
    return 0;

  unsigned Mask = 0;
  std::istringstream Stream(EnvValue);
  std::string Token;
  while (std::getline(Stream, Token, ';')) {
    if (Token.empty())
      continue;
    if (Token == "specialization")
      Mask |= static_cast<unsigned>(TraceOption::Specialization);
    else if (Token == "ir-dump")
      Mask |= static_cast<unsigned>(TraceOption::IRDump);
    else if (Token == "kernel-trace")
      Mask |= static_cast<unsigned>(TraceOption::KernelTrace);
    else
      reportFatalError("Unknown PROTEUS_TRACE_OUTPUT token: " + Token);
  }
  return Mask;
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
  static constexpr bool DefaultSpecializeDimsRange =
#if PROTEUS_ENABLE_CUDA
      // Disable SpecializeDimsRange on CUDA builds: empirically causes worse
      // optimization, so default to false.
      false;
#else
      true;
#endif

  static CodegenOption getCodeGen(CodegenOption ProteusCodegen) {
    constexpr bool SupportOnlyRTC =
#if defined(PROTEUS_ENABLE_CUDA)
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
                       int MinBlocksPerSM = 0)
      : ProteusOptPipeline(ProteusOptPipeline), ProteusCodegen(ProteusCodegen),
        ProteusSpecializeArgs(ProteusSpecializeArgs),
        ProteusSpecializeLaunchBounds(ProteusSpecializeLaunchBounds),
        ProteusSpecializeDims(ProteusSpecializeDims),
        ProteusSpecializeDimsRange(ProteusSpecializeDimsRange),
        ProteusOptLevel(ProteusOptLevel),
        ProteusCodeGenOptLevel(ProteusCodeGenOptLevel),
        TunedMaxThreads(TunedMaxThreads), MinBlocksPerSM(MinBlocksPerSM) {}

public:
  static CodeGenerationConfig createFromEnv() {
    return CodeGenerationConfig(
        getEnvOrDefaultString("PROTEUS_OPT_PIPELINE"),
        getCodeGen(getEnvOrDefaultCG("PROTEUS_CODEGEN", CodegenOption::RTC)),
        getEnvOrDefaultBool("PROTEUS_SPECIALIZE_ARGS", true),
        getEnvOrDefaultBool("PROTEUS_SPECIALIZE_LAUNCH_BOUNDS", true),
        getEnvOrDefaultBool("PROTEUS_SPECIALIZE_DIMS", true),
        getEnvOrDefaultBool("PROTEUS_SPECIALIZE_DIMS_RANGE",
                            DefaultSpecializeDimsRange),
        getEnvOrDefaultChar("PROTEUS_OPT_LEVEL", '3'),
        getEnvOrDefaultInt("PROTEUS_CODEGEN_OPT_LEVEL", 3));
  }

  static CodeGenerationConfig
  createFromJSONEntry(const llvm::json::Object &Config) {
    auto Pipeline = Config.getString("Pipeline");
    std::optional<std::string> ProteusPipeline;
    if (Pipeline)
      ProteusPipeline = Pipeline.value().str();

    return CodeGenerationConfig(
        ProteusPipeline,
        getCodeGen(
            strToCG(getDefaultValueFromOptional(Config.getString("CodeGen"),
                                                llvm::StringRef("rtc"))
                        .str())),
        getDefaultValueFromOptional(Config.getBoolean("SpecializeArgs"), true),
        getDefaultValueFromOptional(Config.getBoolean("LaunchBounds"), true),
        getDefaultValueFromOptional(Config.getBoolean("SpecializeDims"), true),
        getDefaultValueFromOptional(Config.getBoolean("SpecializeDimsRange"),
                                    DefaultSpecializeDimsRange),
        getDefaultValueFromOptional(Config.getString("OptLevel"),
                                    llvm::StringRef("3"))[0],
        getDefaultValueFromOptional(Config.getInteger("CodeGenOptLevel"),
                                    static_cast<int64_t>(3)),
        getDefaultValueFromOptional(Config.getInteger("TunedMaxThreads"),
                                    static_cast<int64_t>(-1L)),
        getDefaultValueFromOptional(Config.getInteger("MinBlocksPerSM"),
                                    static_cast<int64_t>(0L)));
  }

  CodegenOption codeGenOption() const { return ProteusCodegen; }
  bool specializeArgs() const { return ProteusSpecializeArgs; }
  bool specializeDims() const { return ProteusSpecializeDims; }
  bool specializeDimsRange() const { return ProteusSpecializeDimsRange; }
  bool specializeLaunchBounds() const { return ProteusSpecializeLaunchBounds; }
  char optLevel() const { return ProteusOptLevel; }
  int codeGenOptLevel() const { return ProteusCodeGenOptLevel; }
  std::optional<const std::string> optPipeline() const {
    return ProteusOptPipeline;
  }

  int minBlocksPerSM(int MaxThreads) const {
    // NOTE: We only return the tuned value when the current LBMaxThreads is
    // equal to the tuned one. otherwise we return 0. Not doing so, may result
    // in violating constraints in cases in which LBMaxThreads >
    // TunedMaxThreads
    if (TunedMaxThreads != MaxThreads)
      return 0;
    return MinBlocksPerSM;
  }

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
      reportFatalError("Error when opening json file: " +
                       JSONBuf.getError().message() + "\n");

    llvm::json::Value JsonInfo =
        llvm::cantFail(llvm::json::parse(JSONBuf.get()->getBuffer()),
                       "Cannot convert buffer to json value");

    if (auto *Obj = JsonInfo.getAsObject())
      return *Obj;
    return std::nullopt;
  }();

  if (!JSONRoot)
    reportFatalError("Top-level JSON is not an object.\n");

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
  static Config &get() {
    static Config Conf;
    return Conf;
  }
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
  unsigned ProteusTraceConfig;
  bool ProteusDebugOutput;
  std::optional<const std::string> ProteusCacheDir;
  std::string ProteusObjectCacheChain;
  bool ProteusEnableTimeTrace;
  std::string ProteusTimeTraceFile;
  int ProteusCommThreadPollMs;

  const CodeGenerationConfig &getCGConfig(llvm::StringRef KName = "") const {

    if (TunedConfigs.empty())
      return GlobalCodeGenConfig;

    if (auto It = TunedConfigs.find(KName); It != TunedConfigs.end())
      return It->second;

    return GlobalCodeGenConfig;
  }

  bool traceSpecializations() const {
    return ProteusTraceConfig &
           static_cast<unsigned>(TraceOption::Specialization);
  }
  bool traceIRDump() const {
    return ProteusTraceConfig & static_cast<unsigned>(TraceOption::IRDump);
  }
  bool traceKernels() const {
    return ProteusTraceConfig & static_cast<unsigned>(TraceOption::KernelTrace);
  }

  void dump(llvm::raw_ostream &OS) const {
    auto PrintOut = [](llvm::StringRef ID,
                       const CodeGenerationConfig &KConfig) {
      llvm::SmallString<128> S;
      llvm::raw_svector_ostream OS(S);
      OS << "ID:" << ID << " ";
      KConfig.dump(OS);
      return S;
    };

    OS << "PROTEUS_USE_STORED_CACHE " << ProteusUseStoredCache << "\n";
    OS << "PROTEUS_CACHE_DIR " << Config::get().ProteusCacheDir << "\n";

    OS << PrintOut("Default", GlobalCodeGenConfig) << "\n";
    for (auto &KV : TunedConfigs) {
      OS << PrintOut(KV.getKey(), KV.second) << "\n";
    }
  }

private:
  Config()
      : GlobalCodeGenConfig(CodeGenerationConfig::createFromEnv()),
        TunedConfigs(
            parseJSONConfig(getEnvOrDefaultString("PROTEUS_TUNED_KERNELS"))),
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
    ProteusTraceConfig = parseTraceConfig("PROTEUS_TRACE_OUTPUT");
    ProteusDebugOutput = getEnvOrDefaultBool("PROTEUS_DEBUG_OUTPUT", false);
    ProteusObjectCacheChain =
        getEnvOrDefaultString("PROTEUS_OBJECT_CACHE_CHAIN").value_or("storage");
    ProteusEnableTimeTrace =
        getEnvOrDefaultBool("PROTEUS_ENABLE_TIME_TRACE", false);
    ProteusTimeTraceFile =
        getEnvOrDefaultString("PROTEUS_TIME_TRACE_FILE").value_or("");
    ProteusCommThreadPollMs =
        getEnvOrDefaultInt("PROTEUS_COMM_THREAD_POLL_MS", 50);
  }
};
} // namespace proteus

#endif
