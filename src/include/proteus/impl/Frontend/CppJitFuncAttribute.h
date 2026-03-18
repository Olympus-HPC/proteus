#ifndef PROTEUS_RUNTIME_FRONTEND_CPPJITFUNCATTRIBUTE_H
#define PROTEUS_RUNTIME_FRONTEND_CPPJITFUNCATTRIBUTE_H

#include "proteus/CppJitFuncAttribute.h"
#include "proteus/Frontend/TargetModel.h"

namespace proteus {

void setFuncAttribute(TargetModelType TargetModel, void *KernelFunc,
                      CppJitFuncAttribute Attr, int Value);

} // namespace proteus

#endif
