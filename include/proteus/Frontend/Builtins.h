#ifndef PROTEUS_FRONTEND_BUILTINS_H
#define PROTEUS_FRONTEND_BUILTINS_H

#include "proteus/Frontend/Func.h"

namespace proteus {
namespace builtins {
namespace gpu {

Var<unsigned int> getThreadIdX(FuncBase &Fn);
Var<unsigned int> getThreadIdY(FuncBase &Fn);
Var<unsigned int> getThreadIdZ(FuncBase &Fn);

Var<unsigned int> getBlockIdX(FuncBase &Fn);
Var<unsigned int> getBlockIdY(FuncBase &Fn);
Var<unsigned int> getBlockIdZ(FuncBase &Fn);

Var<unsigned int> getBlockDimX(FuncBase &Fn);
Var<unsigned int> getBlockDimY(FuncBase &Fn);
Var<unsigned int> getBlockDimZ(FuncBase &Fn);

Var<unsigned int> getGridDimX(FuncBase &Fn);
Var<unsigned int> getGridDimY(FuncBase &Fn);
Var<unsigned int> getGridDimZ(FuncBase &Fn);

void syncThreads(FuncBase &Fn);

} // namespace gpu
} // namespace builtins
} // namespace proteus

#endif
