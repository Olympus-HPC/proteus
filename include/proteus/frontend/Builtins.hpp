#ifndef PROTEUS_FRONTEND_BUILTINS_HPP
#define PROTEUS_FRONTEND_BUILTINS_HPP

#include "proteus/frontend/Func.hpp"

namespace proteus {
namespace builtins {

namespace cuda {
inline Var &getThreadIdX(Func &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();

  Var &Ret = Fn.declVarInternal("threadid.x", TypeMap<int>::get(Ctx));

  auto &IRB = Fn.getIRB();
  FunctionCallee Callee = M.getOrInsertFunction("llvm.nvvm.read.ptx.sreg.tid.x",
                                                TypeMap<int>::get(Ctx));
  auto *Call = IRB.CreateCall(Callee);
  Ret.storeValue(Call);

  return Ret;
}

} // namespace cuda
} // namespace builtins
} // namespace proteus

#endif