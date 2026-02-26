#include "proteus/Frontend/Builtins.h"
#include "proteus/Frontend/Func.h"

namespace proteus {
namespace builtins {
namespace gpu {

Var<unsigned int> getThreadIdX(FuncBase &Fn) {
  auto &Ctx = Fn.getContext();

  Var<unsigned int> Ret = Fn.declVar<unsigned int>("threadIdx.x");

  auto *Call = Fn.getCodeBuilder().createCall("llvm.nvvm.read.ptx.sreg.tid.x",
                                              TypeMap<unsigned int>::get(Ctx));
  Ret.storeValue(Call);

  return Ret;
}

Var<unsigned int> getBlockIdX(FuncBase &Fn) {
  auto &Ctx = Fn.getContext();

  Var<unsigned int> Ret = Fn.declVar<unsigned int>("blockIdx.x");

  auto *Call = Fn.getCodeBuilder().createCall("llvm.nvvm.read.ptx.sreg.ctaid.x",
                                              TypeMap<unsigned int>::get(Ctx));
  Ret.storeValue(Call);

  return Ret;
}

Var<unsigned int> getBlockDimX(FuncBase &Fn) {
  auto &Ctx = Fn.getContext();

  Var<unsigned int> Ret = Fn.declVar<unsigned int>("blockDim.x");

  auto *Call = Fn.getCodeBuilder().createCall("llvm.nvvm.read.ptx.sreg.ntid.x",
                                              TypeMap<unsigned int>::get(Ctx));
  Ret.storeValue(Call);

  return Ret;
}
Var<unsigned int> getGridDimX(FuncBase &Fn) {
  auto &Ctx = Fn.getContext();

  Var<unsigned int> Ret = Fn.declVar<unsigned int>("gridDim.x");

  auto *Call = Fn.getCodeBuilder().createCall(
      "llvm.nvvm.read.ptx.sreg.nctaid.x", TypeMap<unsigned int>::get(Ctx));
  Ret.storeValue(Call);

  return Ret;
}

Var<unsigned int> getThreadIdY(FuncBase &Fn) {
  auto &Ctx = Fn.getContext();

  Var<unsigned int> Ret = Fn.declVar<unsigned int>("threadIdx.y");

  auto *Call = Fn.getCodeBuilder().createCall("llvm.nvvm.read.ptx.sreg.tid.y",
                                              TypeMap<unsigned int>::get(Ctx));
  Ret.storeValue(Call);

  return Ret;
}

Var<unsigned int> getThreadIdZ(FuncBase &Fn) {
  auto &Ctx = Fn.getContext();

  Var<unsigned int> Ret = Fn.declVar<unsigned int>("threadIdx.z");

  auto *Call = Fn.getCodeBuilder().createCall("llvm.nvvm.read.ptx.sreg.tid.z",
                                              TypeMap<unsigned int>::get(Ctx));
  Ret.storeValue(Call);

  return Ret;
}

Var<unsigned int> getBlockIdY(FuncBase &Fn) {
  auto &Ctx = Fn.getContext();

  Var<unsigned int> Ret = Fn.declVar<unsigned int>("blockIdx.y");

  auto *Call = Fn.getCodeBuilder().createCall("llvm.nvvm.read.ptx.sreg.ctaid.y",
                                              TypeMap<unsigned int>::get(Ctx));
  Ret.storeValue(Call);

  return Ret;
}

Var<unsigned int> getBlockIdZ(FuncBase &Fn) {
  auto &Ctx = Fn.getContext();

  Var<unsigned int> Ret = Fn.declVar<unsigned int>("blockIdx.z");

  auto *Call = Fn.getCodeBuilder().createCall("llvm.nvvm.read.ptx.sreg.ctaid.z",
                                              TypeMap<unsigned int>::get(Ctx));
  Ret.storeValue(Call);

  return Ret;
}

Var<unsigned int> getBlockDimY(FuncBase &Fn) {
  auto &Ctx = Fn.getContext();

  Var<unsigned int> Ret = Fn.declVar<unsigned int>("blockDim.y");

  auto *Call = Fn.getCodeBuilder().createCall("llvm.nvvm.read.ptx.sreg.ntid.y",
                                              TypeMap<unsigned int>::get(Ctx));
  Ret.storeValue(Call);

  return Ret;
}

Var<unsigned int> getBlockDimZ(FuncBase &Fn) {
  auto &Ctx = Fn.getContext();

  Var<unsigned int> Ret = Fn.declVar<unsigned int>("blockDim.z");

  auto *Call = Fn.getCodeBuilder().createCall("llvm.nvvm.read.ptx.sreg.ntid.z",
                                              TypeMap<unsigned int>::get(Ctx));
  Ret.storeValue(Call);

  return Ret;
}

Var<unsigned int> getGridDimY(FuncBase &Fn) {
  auto &Ctx = Fn.getContext();

  Var<unsigned int> Ret = Fn.declVar<unsigned int>("gridDim.y");

  auto *Call = Fn.getCodeBuilder().createCall(
      "llvm.nvvm.read.ptx.sreg.nctaid.y", TypeMap<unsigned int>::get(Ctx));
  Ret.storeValue(Call);

  return Ret;
}

Var<unsigned int> getGridDimZ(FuncBase &Fn) {
  auto &Ctx = Fn.getContext();

  Var<unsigned int> Ret = Fn.declVar<unsigned int>("gridDim.z");

  auto *Call = Fn.getCodeBuilder().createCall(
      "llvm.nvvm.read.ptx.sreg.nctaid.z", TypeMap<unsigned int>::get(Ctx));
  Ret.storeValue(Call);

  return Ret;
}

void syncThreads(FuncBase &Fn) {
  auto &Ctx = Fn.getContext();
  Fn.getCodeBuilder().createCall("llvm.nvvm.barrier0", TypeMap<void>::get(Ctx));
}

} // namespace gpu
} // namespace builtins
} // namespace proteus
