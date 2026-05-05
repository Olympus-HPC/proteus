#include "proteus/KernelMetadata.h"

#include <utility>

namespace proteus::runtime {

KernelMetadata::KernelMetadata(std::string Name, uint64_t StaticHash,
                               std::vector<char> Bitcode,
                               GlobalMetadataMap Globals)
    : Name(std::move(Name)), StaticHash(StaticHash),
      Bitcode(std::move(Bitcode)), Globals(std::move(Globals)) {}

const std::string &KernelMetadata::getName() const { return Name; }

uint64_t KernelMetadata::getStaticHash() const { return StaticHash; }

const std::vector<char> &KernelMetadata::getBitcode() const { return Bitcode; }

const GlobalMetadataMap &KernelMetadata::getGlobals() const { return Globals; }

#if !defined(PROTEUS_ENABLE_CUDA) && !defined(PROTEUS_ENABLE_HIP)
std::optional<KernelMetadata> captureKernelMetadata(const void *) {
  return std::nullopt;
}
#endif

} // namespace proteus::runtime
