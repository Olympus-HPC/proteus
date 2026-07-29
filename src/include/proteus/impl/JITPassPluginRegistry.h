#ifndef PROTEUS_JIT_PASS_PLUGIN_REGISTRY_H
#define PROTEUS_JIT_PASS_PLUGIN_REGISTRY_H

#include "proteus/Init.h"

#include <optional>
#include <string>
#include <vector>

namespace proteus {

struct JITPassPluginConfig {
  std::string Path;
  std::optional<std::string> Pipeline;
  JITPassPluginPosition Position;
  std::string Fingerprint;
};

void registerJITPassPluginImpl(const std::string &PluginPath,
                               std::optional<std::string> PassPipeline,
                               JITPassPluginPosition Position);
void clearJITPassPluginsImpl();
std::vector<JITPassPluginConfig> getJITPassPluginConfigs();

} // namespace proteus

#endif
