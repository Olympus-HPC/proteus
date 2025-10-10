#ifndef PROTEUS_FRONTEND_BUILTINS_HPP
#define PROTEUS_FRONTEND_BUILTINS_HPP

#include "proteus/Frontend/Func.hpp"

namespace proteus {
namespace builtins {
namespace gpu {

VarTT<int> getThreadIdX(FuncBase &Fn);
VarTT<int> getThreadIdY(FuncBase &Fn);
VarTT<int> getThreadIdZ(FuncBase &Fn);

VarTT<int> getBlockIdX(FuncBase &Fn);
VarTT<int> getBlockIdY(FuncBase &Fn);
VarTT<int> getBlockIdZ(FuncBase &Fn);

VarTT<int> getBlockDimX(FuncBase &Fn);
VarTT<int> getBlockDimY(FuncBase &Fn);
VarTT<int> getBlockDimZ(FuncBase &Fn);

VarTT<int> getGridDimX(FuncBase &Fn);
VarTT<int> getGridDimY(FuncBase &Fn);
VarTT<int> getGridDimZ(FuncBase &Fn);

void syncThreads(FuncBase &Fn);

} // namespace gpu
} // namespace builtins
} // namespace proteus

#endif
