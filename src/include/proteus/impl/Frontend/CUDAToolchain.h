#ifndef PROTEUS_RUNTIME_FRONTEND_CUDATOOLCHAIN_H
#define PROTEUS_RUNTIME_FRONTEND_CUDATOOLCHAIN_H

#include <string>

namespace proteus {

struct ResolvedCUDAToolchain {
  std::string Root;
  std::string LibDevicePath;
  std::string RuntimeLibDir;
  std::string Origin;
};

const ResolvedCUDAToolchain &resolveCUDAToolchain();

} // namespace proteus

#endif
