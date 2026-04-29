#ifndef PROTEUS_RUNTIME_FRONTEND_HIPTOOLCHAIN_H
#define PROTEUS_RUNTIME_FRONTEND_HIPTOOLCHAIN_H

#include <string>

namespace proteus {

struct ResolvedHIPToolchain {
  std::string RocmRoot;
  std::string DeviceLibDir;
  std::string RuntimeVersion;
  std::string Origin;
};

const ResolvedHIPToolchain &resolveHIPToolchain();

} // namespace proteus

#endif
