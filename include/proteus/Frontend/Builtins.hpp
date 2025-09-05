#ifndef PROTEUS_FRONTEND_BUILTINS_HPP
#define PROTEUS_FRONTEND_BUILTINS_HPP

#include "proteus/Frontend/Func.hpp"

namespace proteus {
namespace builtins {
namespace gpu {

Var &getThreadIdX(FuncBase &Fn);
Var &getThreadIdY(FuncBase &Fn);
Var &getThreadIdZ(FuncBase &Fn);

Var &getBlockIdX(FuncBase &Fn);
Var &getBlockIdY(FuncBase &Fn);
Var &getBlockIdZ(FuncBase &Fn);

Var &getBlockDimX(FuncBase &Fn);
Var &getBlockDimY(FuncBase &Fn);
Var &getBlockDimZ(FuncBase &Fn);

Var &getGridDimX(FuncBase &Fn);
Var &getGridDimY(FuncBase &Fn);
Var &getGridDimZ(FuncBase &Fn);

} // namespace gpu
} // namespace builtins
} // namespace proteus

#endif
