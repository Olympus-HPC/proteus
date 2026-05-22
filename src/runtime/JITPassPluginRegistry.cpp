#include "proteus/impl/JITPassPluginRegistry.h"

#include "proteus/Error.h"

#include <llvm/ADT/SmallString.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>

#include <algorithm>
#include <mutex>

namespace proteus {
namespace {

class JITPassPluginRegistry {
public:
  static JITPassPluginRegistry &instance() {
    static JITPassPluginRegistry Registry;
    return Registry;
  }

  void registerPlugin(const std::string &PluginPath,
                      const std::string &PassPipeline) {
    if (PluginPath.empty())
      reportFatalError("JIT pass plugin path must be non-empty");
    if (PassPipeline.empty())
      reportFatalError("JIT pass plugin pipeline must be non-empty");

    JITPassPluginConfig Config{normalizePath(PluginPath), PassPipeline, {}};
    Config.Fingerprint = computeFingerprint(Config);

    std::lock_guard<std::mutex> Lock(Mutex);
    auto It = std::find_if(
        Plugins.begin(), Plugins.end(), [&](const JITPassPluginConfig &Entry) {
          return Entry.Path == Config.Path && Entry.Pipeline == Config.Pipeline;
        });
    if (It != Plugins.end()) {
      It->Fingerprint = std::move(Config.Fingerprint);
      return;
    }

    Plugins.push_back(std::move(Config));
  }

  void clear() {
    std::lock_guard<std::mutex> Lock(Mutex);
    Plugins.clear();
  }

  std::vector<JITPassPluginConfig> snapshot() {
    std::lock_guard<std::mutex> Lock(Mutex);
    return Plugins;
  }

private:
  static std::string normalizePath(const std::string &PluginPath) {
    llvm::SmallString<256> RealPath;
    if (!llvm::sys::fs::real_path(PluginPath, RealPath))
      return std::string(RealPath.str());

    return PluginPath;
  }

  static std::string computeFingerprint(const JITPassPluginConfig &Config) {
    llvm::sys::fs::file_status Status;
    if (llvm::sys::fs::status(Config.Path, Status))
      return Config.Path + "|" + Config.Pipeline;

    llvm::SmallString<128> Storage;
    llvm::raw_svector_ostream OS(Storage);
    const llvm::sys::fs::UniqueID ID = Status.getUniqueID();
    OS << Config.Path << "|" << Config.Pipeline << "|" << ID.getDevice() << ":"
       << ID.getFile() << "|" << Status.getSize() << "|"
       << Status.getLastModificationTime().time_since_epoch().count();
    return std::string(OS.str());
  }

  std::mutex Mutex;
  std::vector<JITPassPluginConfig> Plugins;
};

} // namespace

void registerJITPassPluginImpl(const std::string &PluginPath,
                               const std::string &PassPipeline) {
  JITPassPluginRegistry::instance().registerPlugin(PluginPath, PassPipeline);
}

void clearJITPassPluginsImpl() { JITPassPluginRegistry::instance().clear(); }

std::vector<JITPassPluginConfig> getJITPassPluginConfigs() {
  return JITPassPluginRegistry::instance().snapshot();
}

} // namespace proteus
