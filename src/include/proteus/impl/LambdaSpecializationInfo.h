#ifndef PROTEUS_LAMBDA_SPECIALIZATION_INFO_H
#define PROTEUS_LAMBDA_SPECIALIZATION_INFO_H

#include "proteus/CompilerInterfaceTypes.h"

#include <llvm/ADT/SmallVector.h>

#include <cstdint>
#include <string>

namespace proteus {

struct LambdaCalleeInfo {
  std::string CalleeName;
  std::string LambdaType;
  int32_t KernelArgIndex = -1;
};

struct ResolvedLambdaSpecializationInfo {
  std::string CalleeName;
  llvm::SmallVector<RuntimeConstant> Values;
};

} // namespace proteus

#endif
