#ifndef PROTEUS_CONFIG_HPP
#define PROTEUS_CONFIG_HPP

#include <string>

namespace proteus {

inline bool getEnvOrDefaultBool(const char *VarName, bool Default) {

  const char *EnvValue = std::getenv(VarName);
  return EnvValue ? static_cast<bool>(std::stoi(EnvValue)) : Default;
}

inline int getEnvOrDefaultInt(const char *VarName, int Default) {

  const char *EnvValue = std::getenv(VarName);
  return EnvValue ? std::stoi(EnvValue) : Default;
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
  bool ProteusUseHIPRTCCodegen;
  bool ProteusDisable;
  bool ProteusDumpLLVMIR;
  bool ProteusRelinkGlobalsByCopy;
  bool ProteusAsyncCompilation;
  int ProteusAsyncThreads;
  bool ProteusAsyncTestBlocking;
  bool ProteusUseLightweightKernelClone;
  bool ProteusEnableTimers;

private:
  Config() {
    ProteusUseStoredCache =
        getEnvOrDefaultBool("PROTEUS_USE_STORED_CACHE", true);
    ProteusSpecializeLaunchBounds =
        getEnvOrDefaultBool("PROTEUS_SPECIALIZE_LAUNCH_BOUNDS", true);
    ProteusSpecializeArgs =
        getEnvOrDefaultBool("PROTEUS_SPECIALIZE_ARGS", true);
    ProteusSpecializeDims =
        getEnvOrDefaultBool("PROTEUS_SPECIALIZE_DIMS", true);
    ProteusUseHIPRTCCodegen =
        getEnvOrDefaultBool("PROTEUS_USE_HIP_RTC_CODEGEN", true);
    ProteusDisable = getEnvOrDefaultBool("PROTEUS_DISABLE", false);
    ProteusDumpLLVMIR = getEnvOrDefaultBool("PROTEUS_DUMP_LLVM_IR", false);
    ProteusRelinkGlobalsByCopy =
        getEnvOrDefaultBool("PROTEUS_RELINK_GLOBALS_BY_COPY", false);
    ProteusAsyncCompilation =
        getEnvOrDefaultBool("PROTEUS_ASYNC_COMPILATION", false);
    ProteusAsyncTestBlocking =
        getEnvOrDefaultBool("PROTEUS_ASYNC_TEST_BLOCKING", false);
    ProteusAsyncThreads = getEnvOrDefaultInt("PROTEUS_ASYNC_THREADS", 1);
    ProteusUseLightweightKernelClone =
        getEnvOrDefaultBool("PROTEUS_USE_LIGHTWEIGHT_KERNEL_CLONE", true);
    ProteusEnableTimers = getEnvOrDefaultBool("PROTEUS_ENABLE_TIMERS", false);
  }
};
} // namespace proteus

#endif
