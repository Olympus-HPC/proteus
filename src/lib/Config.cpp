#include "proteus/Config.hpp"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

namespace proteus {

const bool CodeGenerationConfig::DefaultSpecializeDimsRange =
    !PROTEUS_ENABLE_CUDA;

CodeGenerationConfig::CodeGenerationConfig(
    std::optional<const std::string> ProteusOptPipeline,
    CodegenOption ProteusCodegen, bool ProteusSpecializeArgs,
    bool ProteusSpecializeLaunchBounds, bool ProteusSpecializeDims,
    bool ProteusSpecializeDimsRange, char ProteusOptLevel,
    int ProteusCodeGenOptLevel, int TunedMaxThreads, int MinBlocksPerSM)
    : ProteusOptPipeline(ProteusOptPipeline), ProteusCodegen(ProteusCodegen),
      ProteusSpecializeArgs(ProteusSpecializeArgs),
      ProteusSpecializeLaunchBounds(ProteusSpecializeLaunchBounds),
      ProteusSpecializeDims(ProteusSpecializeDims),
      ProteusSpecializeDimsRange(ProteusSpecializeDimsRange),
      ProteusOptLevel(ProteusOptLevel),
      ProteusCodeGenOptLevel(ProteusCodeGenOptLevel),
      TunedMaxThreads(TunedMaxThreads), MinBlocksPerSM(MinBlocksPerSM) {}

CodegenOption CodeGenerationConfig::getCodeGen(CodegenOption ProteusCodegen) {
  constexpr bool SupportOnlyRTC = PROTEUS_ENABLE_CUDA;
  if (SupportOnlyRTC && ProteusCodegen != CodegenOption::RTC) {
    Logger::outs("proteus") << "Warning: Proteus supports only RTC in the "
                               "current build system configuration, "
                               "defaulting Codegen to RTC\n";
    ProteusCodegen = CodegenOption::RTC;
  }
  return ProteusCodegen;
}

CodeGenerationConfig CodeGenerationConfig::createFromEnv() {
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

CodeGenerationConfig
CodeGenerationConfig::createFromJSONEntry(const llvm::json::Object &Config) {

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

CodegenOption CodeGenerationConfig::codeGenOption() const {
  return ProteusCodegen;
}

bool CodeGenerationConfig::specializeArgs() const {
  return ProteusSpecializeArgs;
}

bool CodeGenerationConfig::specializeDims() const {
  return ProteusSpecializeDims;
}

bool CodeGenerationConfig::specializeDimsRange() const {
  return ProteusSpecializeDimsRange;
}

bool CodeGenerationConfig::specializeLaunchBounds() const {
  return ProteusSpecializeLaunchBounds;
}

char CodeGenerationConfig::optLevel() const { return ProteusOptLevel; }

int CodeGenerationConfig::codeGenOptLevel() const {
  return ProteusCodeGenOptLevel;
}

std::optional<const std::string> CodeGenerationConfig::optPipeline() const {
  return ProteusOptPipeline;
}

int CodeGenerationConfig::minBlocksPerSM(int MaxThreads) const {
  // NOTE: We only return the tuned value when the current LBMaxThreads is equal
  // to the tuned one. Otherwise we return 0. Not doing so may result in
  // violating constraints in cases in which LBMaxThreads > TunedMaxThreads.
  if (TunedMaxThreads != MaxThreads)
    return 0;
  return MinBlocksPerSM;
}

Config &Config::get() {
  static Config Conf;
  return Conf;
}

const CodeGenerationConfig &Config::getCGConfig(llvm::StringRef KName) const {
  if (TunedConfigs.empty())
    return GlobalCodeGenConfig;

  if (auto It = TunedConfigs.find(KName); It != TunedConfigs.end())
    return It->second;

  return GlobalCodeGenConfig;
}

void Config::dump(llvm::raw_ostream &OS) const {
  auto PrintOut = [](llvm::StringRef ID, const CodeGenerationConfig &KConfig) {
    llvm::SmallString<128> S;
    llvm::raw_svector_ostream Buffer(S);
    Buffer << "ID:" << ID << " ";
    KConfig.dump(Buffer);
    return S;
  };

  OS << "PROTEUS_USE_STORED_CACHE " << ProteusUseStoredCache << "\n";
  OS << "PROTEUS_CACHE_DIR " << Config::get().ProteusCacheDir << "\n";

  OS << PrintOut("Default", GlobalCodeGenConfig) << "\n";
  for (auto &KV : TunedConfigs) {
    OS << PrintOut(KV.getKey(), KV.second) << "\n";
  }
}

Config::Config()
    : GlobalCodeGenConfig(CodeGenerationConfig::createFromEnv()),
      TunedConfigs(
          parseJSONConfig(getEnvOrDefaultString("PROTEUS_TUNED_KERNELS"))),
      ProteusCacheDir(getEnvOrDefaultString("PROTEUS_CACHE_DIR")) {
  ProteusUseStoredCache = getEnvOrDefaultBool("PROTEUS_USE_STORED_CACHE", true);
  ProteusDisable = getEnvOrDefaultBool("PROTEUS_DISABLE", false);
  ProteusDumpLLVMIR = getEnvOrDefaultBool("PROTEUS_DUMP_LLVM_IR", false);
  ProteusRelinkGlobalsByCopy =
      getEnvOrDefaultBool("PROTEUS_RELINK_GLOBALS_BY_COPY", false);
  ProteusAsyncCompilation =
      getEnvOrDefaultBool("PROTEUS_ASYNC_COMPILATION", false);
  ProteusAsyncTestBlocking =
      getEnvOrDefaultBool("PROTEUS_ASYNC_TEST_BLOCKING", false);
  ProteusAsyncThreads = getEnvOrDefaultInt("PROTEUS_ASYNC_THREADS", 1);
  ProteusKernelClone =
      getEnvOrDefaultKC("PROTEUS_KERNEL_CLONE", KernelCloneOption::CrossClone);
  ProteusEnableTimers = getEnvOrDefaultBool("PROTEUS_ENABLE_TIMERS", false);
  ProteusTraceOutput = getEnvOrDefaultInt("PROTEUS_TRACE_OUTPUT", 0);
  ProteusDebugOutput = getEnvOrDefaultBool("PROTEUS_DEBUG_OUTPUT", false);
}

} // namespace proteus
