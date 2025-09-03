#ifndef PROTEUS_TARGET_MODE_H
#define PROTEUS_TARGET_MODE_H

#include <llvm/ADT/StringRef.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/TargetParser/Triple.h>

#include "proteus/Error.h"

namespace proteus {

using namespace llvm;

enum class TargetModelType { HOST, CUDA, HIP, HOST_HIP, HOST_CUDA };

inline TargetModelType parseTargetModel(StringRef Target) {
  if (Target == "host" || Target == "native") {
    return TargetModelType::HOST;
  }

  if (Target == "cuda") {
    return TargetModelType::CUDA;
  }

  if (Target == "hip") {
    return TargetModelType::HIP;
  }

  if (Target == "host_hip") {
    return TargetModelType::HOST_HIP;
  }

  if (Target == "host_cuda") {
    return TargetModelType::HOST_CUDA;
  }

  PROTEUS_FATAL_ERROR("Unsupported target " + Target);
}

inline std::string getTargetTriple(TargetModelType Model) {
  switch (Model) {
  case TargetModelType::HOST_HIP:
  case TargetModelType::HOST_CUDA:
  case TargetModelType::HOST:
    return sys::getProcessTriple();
  case TargetModelType::CUDA:
    return "nvptx64-nvidia-cuda";
  case TargetModelType::HIP:
    return "amdgcn-amd-amdhsa";
  default:
    PROTEUS_FATAL_ERROR("Unsupported target model");
  }
}

inline bool isHostTargetModel(TargetModelType TargetModel) {
  return (TargetModel == TargetModelType::HOST) ||
         (TargetModel == TargetModelType::HOST_HIP) ||
         (TargetModel == TargetModelType::HOST_CUDA);
}

} // namespace proteus

#endif
