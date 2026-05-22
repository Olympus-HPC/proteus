#ifndef PROTEUS_JIT_PASS_PLUGIN_REGISTRY_H
#define PROTEUS_JIT_PASS_PLUGIN_REGISTRY_H

#include <string>
#include <vector>

namespace proteus {

struct JITPassPluginConfig {
  std::string Path;
  std::string Pipeline;
  std::string Fingerprint;
};

void registerJITPassPluginImpl(const std::string &PluginPath,
                               const std::string &PassPipeline);
void clearJITPassPluginsImpl();
std::vector<JITPassPluginConfig> getJITPassPluginConfigs();

} // namespace proteus

#endif
