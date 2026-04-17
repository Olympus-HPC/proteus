#ifndef PROTEUS_RUNTIME_FRONTEND_JITFUNCATTRIBUTE_H
#define PROTEUS_RUNTIME_FRONTEND_JITFUNCATTRIBUTE_H

#include "proteus/Frontend/TargetModel.h"
#include "proteus/JitFuncAttribute.h"

namespace proteus {

void setFuncAttribute(TargetModelType TargetModel, void *KernelFunc,
                      JitFuncAttribute Attr, int Value);

} // namespace proteus

#endif
