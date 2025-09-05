#include "proteus/Frontend/Builtins.hpp"
#include "proteus/Frontend/Func.hpp"

namespace proteus {
namespace builtins {
namespace gpu {

Var &getThreadIdX(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();

  Var &Ret = Fn.declVarInternal("threadIdx.x", TypeMap<int>::get(Ctx));

  auto &IRB = Fn.getIRBuilder();
  FunctionCallee Callee = M.getOrInsertFunction("llvm.nvvm.read.ptx.sreg.tid.x",
                                                TypeMap<int>::get(Ctx));
  auto *Call = IRB.CreateCall(Callee);
  Ret.storeValue(Call);

  return Ret;
}

Var &getBlockIdX(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();

  Var &Ret = Fn.declVarInternal("blockIdx.x", TypeMap<int>::get(Ctx));

  auto &IRB = Fn.getIRBuilder();
  FunctionCallee Callee = M.getOrInsertFunction(
      "llvm.nvvm.read.ptx.sreg.ctaid.x", TypeMap<int>::get(Ctx));
  auto *Call = IRB.CreateCall(Callee);
  Ret.storeValue(Call);

  return Ret;
}

Var &getBlockDimX(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();

  Var &Ret = Fn.declVarInternal("blockDim.x", TypeMap<int>::get(Ctx));

  auto &IRB = Fn.getIRBuilder();
  FunctionCallee Callee = M.getOrInsertFunction(
      "llvm.nvvm.read.ptx.sreg.ntid.x", TypeMap<int>::get(Ctx));
  auto *Call = IRB.CreateCall(Callee);
  Ret.storeValue(Call);

  return Ret;
}

Var &getGridDimX(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();

  Var &Ret = Fn.declVarInternal("gridDim.x", TypeMap<int>::get(Ctx));

  auto &IRB = Fn.getIRBuilder();
  FunctionCallee Callee = M.getOrInsertFunction(
      "llvm.nvvm.read.ptx.sreg.nctaid.x", TypeMap<int>::get(Ctx));
  auto *Call = IRB.CreateCall(Callee);
  Ret.storeValue(Call);

  return Ret;
}

Var &getThreadIdY(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();

  Var &Ret = Fn.declVarInternal("threadIdx.y", TypeMap<int>::get(Ctx));

  auto &IRB = Fn.getIRBuilder();
  FunctionCallee Callee = M.getOrInsertFunction("llvm.nvvm.read.ptx.sreg.tid.y",
                                                TypeMap<int>::get(Ctx));
  auto *Call = IRB.CreateCall(Callee);
  Ret.storeValue(Call);

  return Ret;
}

Var &getThreadIdZ(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();

  Var &Ret = Fn.declVarInternal("threadIdx.z", TypeMap<int>::get(Ctx));

  auto &IRB = Fn.getIRBuilder();
  FunctionCallee Callee = M.getOrInsertFunction("llvm.nvvm.read.ptx.sreg.tid.z",
                                                TypeMap<int>::get(Ctx));
  auto *Call = IRB.CreateCall(Callee);
  Ret.storeValue(Call);

  return Ret;
}

Var &getBlockIdY(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();

  Var &Ret = Fn.declVarInternal("blockIdx.y", TypeMap<int>::get(Ctx));

  auto &IRB = Fn.getIRBuilder();
  FunctionCallee Callee = M.getOrInsertFunction(
      "llvm.nvvm.read.ptx.sreg.ctaid.y", TypeMap<int>::get(Ctx));
  auto *Call = IRB.CreateCall(Callee);
  Ret.storeValue(Call);

  return Ret;
}

Var &getBlockIdZ(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();

  Var &Ret = Fn.declVarInternal("blockIdx.z", TypeMap<int>::get(Ctx));

  auto &IRB = Fn.getIRBuilder();
  FunctionCallee Callee = M.getOrInsertFunction(
      "llvm.nvvm.read.ptx.sreg.ctaid.z", TypeMap<int>::get(Ctx));
  auto *Call = IRB.CreateCall(Callee);
  Ret.storeValue(Call);

  return Ret;
}

Var &getBlockDimY(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();

  Var &Ret = Fn.declVarInternal("blockDim.y", TypeMap<int>::get(Ctx));

  auto &IRB = Fn.getIRBuilder();
  FunctionCallee Callee = M.getOrInsertFunction(
      "llvm.nvvm.read.ptx.sreg.ntid.y", TypeMap<int>::get(Ctx));
  auto *Call = IRB.CreateCall(Callee);
  Ret.storeValue(Call);

  return Ret;
}

Var &getBlockDimZ(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();

  Var &Ret = Fn.declVarInternal("blockDim.z", TypeMap<int>::get(Ctx));

  auto &IRB = Fn.getIRBuilder();
  FunctionCallee Callee = M.getOrInsertFunction(
      "llvm.nvvm.read.ptx.sreg.ntid.z", TypeMap<int>::get(Ctx));
  auto *Call = IRB.CreateCall(Callee);
  Ret.storeValue(Call);

  return Ret;
}

Var &getGridDimY(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();

  Var &Ret = Fn.declVarInternal("gridDim.y", TypeMap<int>::get(Ctx));

  auto &IRB = Fn.getIRBuilder();
  FunctionCallee Callee = M.getOrInsertFunction(
      "llvm.nvvm.read.ptx.sreg.nctaid.y", TypeMap<int>::get(Ctx));
  auto *Call = IRB.CreateCall(Callee);
  Ret.storeValue(Call);

  return Ret;
}

Var &getGridDimZ(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();

  Var &Ret = Fn.declVarInternal("gridDim.z", TypeMap<int>::get(Ctx));

  auto &IRB = Fn.getIRBuilder();
  FunctionCallee Callee = M.getOrInsertFunction(
      "llvm.nvvm.read.ptx.sreg.nctaid.z", TypeMap<int>::get(Ctx));
  auto *Call = IRB.CreateCall(Callee);
  Ret.storeValue(Call);

  return Ret;
}

} // namespace gpu
} // namespace builtins
} // namespace proteus
