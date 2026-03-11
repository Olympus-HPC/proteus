#include "proteus/Frontend/Builtins.h"
#include "proteus/Frontend/Func.h"

namespace proteus {
namespace builtins {
namespace gpu {

namespace {

template <typename RetT>
Var<RetT> emitBuiltinVar(FuncBase &Fn, const std::string &BuiltinName,
                         const std::string &VarName) {
  Var<RetT> Ret = Fn.declVar<RetT>(VarName);
  IRType RetTy = TypeMap<RetT>::get();
  IRValue *V = Fn.getCodeBuilder().emitBuiltin(BuiltinName, RetTy, {});
  Ret.storeValue(V);
  return Ret;
}

} // namespace

Var<unsigned int> getThreadIdX(FuncBase &Fn) {
  return emitBuiltinVar<unsigned int>(Fn, "threadIdx.x", "threadIdx.x");
}

Var<unsigned int> getThreadIdY(FuncBase &Fn) {
  return emitBuiltinVar<unsigned int>(Fn, "threadIdx.y", "threadIdx.y");
}

Var<unsigned int> getThreadIdZ(FuncBase &Fn) {
  return emitBuiltinVar<unsigned int>(Fn, "threadIdx.z", "threadIdx.z");
}

Var<unsigned int> getBlockIdX(FuncBase &Fn) {
  return emitBuiltinVar<unsigned int>(Fn, "blockIdx.x", "blockIdx.x");
}

Var<unsigned int> getBlockIdY(FuncBase &Fn) {
  return emitBuiltinVar<unsigned int>(Fn, "blockIdx.y", "blockIdx.y");
}

Var<unsigned int> getBlockIdZ(FuncBase &Fn) {
  return emitBuiltinVar<unsigned int>(Fn, "blockIdx.z", "blockIdx.z");
}

Var<unsigned int> getBlockDimX(FuncBase &Fn) {
  return emitBuiltinVar<unsigned int>(Fn, "blockDim.x", "blockDim.x");
}

Var<unsigned int> getBlockDimY(FuncBase &Fn) {
  return emitBuiltinVar<unsigned int>(Fn, "blockDim.y", "blockDim.y");
}

Var<unsigned int> getBlockDimZ(FuncBase &Fn) {
  return emitBuiltinVar<unsigned int>(Fn, "blockDim.z", "blockDim.z");
}

Var<unsigned int> getGridDimX(FuncBase &Fn) {
  return emitBuiltinVar<unsigned int>(Fn, "gridDim.x", "gridDim.x");
}

Var<unsigned int> getGridDimY(FuncBase &Fn) {
  return emitBuiltinVar<unsigned int>(Fn, "gridDim.y", "gridDim.y");
}

Var<unsigned int> getGridDimZ(FuncBase &Fn) {
  return emitBuiltinVar<unsigned int>(Fn, "gridDim.z", "gridDim.z");
}

void syncThreads(FuncBase &Fn) {
  Fn.getCodeBuilder().emitBuiltin("syncThreads", TypeMap<void>::get(), {});
}

} // namespace gpu
} // namespace builtins
} // namespace proteus
