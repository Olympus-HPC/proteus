#ifndef PROTEUS_TARGET_MODE_H
#define PROTEUS_TARGET_MODE_H

#include <string>

namespace proteus {

enum class TargetModelType { HOST, CUDA, HIP, HOST_HIP, HOST_CUDA };

TargetModelType parseTargetModel(const std::string &Target);

std::string getTargetTriple(TargetModelType Model);

bool isHostTargetModel(TargetModelType TargetModel);

} // namespace proteus

#endif
