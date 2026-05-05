#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace proteus::runtime {

struct GlobalMetadata {
  const void *HostAddr = nullptr;
  const void *DevAddr = nullptr;
  uint64_t VarSize = 0;
};

using GlobalMetadataMap = std::unordered_map<std::string, GlobalMetadata>;

class KernelMetadata {
public:
  KernelMetadata(std::string Name, uint64_t StaticHash,
                 std::vector<char> Bitcode, GlobalMetadataMap Globals);

  const std::string &getName() const;
  uint64_t getStaticHash() const;
  const std::vector<char> &getBitcode() const;
  const GlobalMetadataMap &getGlobals() const;

private:
  std::string Name;
  uint64_t StaticHash = 0;
  std::vector<char> Bitcode;
  GlobalMetadataMap Globals;
};

std::optional<KernelMetadata> captureKernelMetadata(const void *Kernel);

} // namespace proteus::runtime
