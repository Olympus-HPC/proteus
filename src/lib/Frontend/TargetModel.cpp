#include "proteus/Frontend/TargetModel.h"
#include "proteus/Error.h"

#include <llvm/TargetParser/Host.h>

namespace proteus {

using namespace llvm;

TargetModelType parseTargetModel(const std::string &Target) {
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

  reportFatalError("Unsupported target " + Target);
}

std::string getTargetTriple(TargetModelType Model) {
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
    reportFatalError("Unsupported target model");
  }
}

bool isHostTargetModel(TargetModelType TargetModel) {
  return (TargetModel == TargetModelType::HOST) ||
         (TargetModel == TargetModelType::HOST_HIP) ||
         (TargetModel == TargetModelType::HOST_CUDA);
}

} // namespace proteus
