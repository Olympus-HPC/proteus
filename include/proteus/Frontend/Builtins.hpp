#ifndef PROTEUS_FRONTEND_BUILTINS_HPP
#define PROTEUS_FRONTEND_BUILTINS_HPP

#include "proteus/Frontend/Func.hpp"

namespace proteus {
namespace builtins {
namespace gpu {

Var<int> getThreadIdX(FuncBase &Fn);
Var<int> getThreadIdY(FuncBase &Fn);
Var<int> getThreadIdZ(FuncBase &Fn);

Var<int> getBlockIdX(FuncBase &Fn);
Var<int> getBlockIdY(FuncBase &Fn);
Var<int> getBlockIdZ(FuncBase &Fn);

Var<int> getBlockDimX(FuncBase &Fn);
Var<int> getBlockDimY(FuncBase &Fn);
Var<int> getBlockDimZ(FuncBase &Fn);

Var<int> getGridDimX(FuncBase &Fn);
Var<int> getGridDimY(FuncBase &Fn);
Var<int> getGridDimZ(FuncBase &Fn);

void syncThreads(FuncBase &Fn);

} // namespace gpu
} // namespace builtins
} // namespace proteus

#endif
