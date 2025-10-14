#include "proteus/Frontend/Builtins.hpp"
#include "proteus/Frontend/Func.hpp"

namespace proteus {
namespace builtins {
namespace gpu {

Var<unsigned int> getThreadIdX(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();

  Var<unsigned int> Ret = Fn.declVarInternal<unsigned int>("threadIdx.x");

  auto &IRB = Fn.getIRBuilder();
  FunctionCallee Callee = M.getOrInsertFunction(
      "llvm.nvvm.read.ptx.sreg.tid.x", TypeMap<unsigned int>::get(Ctx));
  auto *Call = IRB.CreateCall(Callee);
  Ret.storeValue(Call);

  return Ret;
}

Var<unsigned int> getBlockIdX(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();

  Var<unsigned int> Ret = Fn.declVarInternal<unsigned int>("blockIdx.x");

  auto &IRB = Fn.getIRBuilder();
  FunctionCallee Callee = M.getOrInsertFunction(
      "llvm.nvvm.read.ptx.sreg.ctaid.x", TypeMap<unsigned int>::get(Ctx));
  auto *Call = IRB.CreateCall(Callee);
  Ret.storeValue(Call);

  return Ret;
}

Var<unsigned int> getBlockDimX(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();

  Var<unsigned int> Ret = Fn.declVarInternal<unsigned int>("blockDim.x");

  auto &IRB = Fn.getIRBuilder();
  FunctionCallee Callee = M.getOrInsertFunction(
      "llvm.nvvm.read.ptx.sreg.ntid.x", TypeMap<unsigned int>::get(Ctx));
  auto *Call = IRB.CreateCall(Callee);
  Ret.storeValue(Call);

  return Ret;
}

Var<unsigned int> getGridDimX(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();

  Var<unsigned int> Ret = Fn.declVarInternal<unsigned int>("gridDim.x");

  auto &IRB = Fn.getIRBuilder();
  FunctionCallee Callee = M.getOrInsertFunction(
      "llvm.nvvm.read.ptx.sreg.nctaid.x", TypeMap<unsigned int>::get(Ctx));
  auto *Call = IRB.CreateCall(Callee);
  Ret.storeValue(Call);

  return Ret;
}

Var<unsigned int> getThreadIdY(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();

  Var<unsigned int> Ret = Fn.declVarInternal<unsigned int>("threadIdx.y");

  auto &IRB = Fn.getIRBuilder();
  FunctionCallee Callee = M.getOrInsertFunction(
      "llvm.nvvm.read.ptx.sreg.tid.y", TypeMap<unsigned int>::get(Ctx));
  auto *Call = IRB.CreateCall(Callee);
  Ret.storeValue(Call);

  return Ret;
}

Var<unsigned int> getThreadIdZ(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();

  Var<unsigned int> Ret = Fn.declVarInternal<unsigned int>("threadIdx.z");

  auto &IRB = Fn.getIRBuilder();
  FunctionCallee Callee = M.getOrInsertFunction(
      "llvm.nvvm.read.ptx.sreg.tid.z", TypeMap<unsigned int>::get(Ctx));
  auto *Call = IRB.CreateCall(Callee);
  Ret.storeValue(Call);

  return Ret;
}

Var<unsigned int> getBlockIdY(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();

  Var<unsigned int> Ret = Fn.declVarInternal<unsigned int>("blockIdx.y");

  auto &IRB = Fn.getIRBuilder();
  FunctionCallee Callee = M.getOrInsertFunction(
      "llvm.nvvm.read.ptx.sreg.ctaid.y", TypeMap<unsigned int>::get(Ctx));
  auto *Call = IRB.CreateCall(Callee);
  Ret.storeValue(Call);

  return Ret;
}

Var<unsigned int> getBlockIdZ(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();

  Var<unsigned int> Ret = Fn.declVarInternal<unsigned int>("blockIdx.z");

  auto &IRB = Fn.getIRBuilder();
  FunctionCallee Callee = M.getOrInsertFunction(
      "llvm.nvvm.read.ptx.sreg.ctaid.z", TypeMap<unsigned int>::get(Ctx));
  auto *Call = IRB.CreateCall(Callee);
  Ret.storeValue(Call);

  return Ret;
}

Var<unsigned int> getBlockDimY(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();

  Var<unsigned int> Ret = Fn.declVarInternal<unsigned int>("blockDim.y");

  auto &IRB = Fn.getIRBuilder();
  FunctionCallee Callee = M.getOrInsertFunction(
      "llvm.nvvm.read.ptx.sreg.ntid.y", TypeMap<unsigned int>::get(Ctx));
  auto *Call = IRB.CreateCall(Callee);
  Ret.storeValue(Call);

  return Ret;
}

Var<unsigned int> getBlockDimZ(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();

  Var<unsigned int> Ret = Fn.declVarInternal<unsigned int>("blockDim.z");

  auto &IRB = Fn.getIRBuilder();
  FunctionCallee Callee = M.getOrInsertFunction(
      "llvm.nvvm.read.ptx.sreg.ntid.z", TypeMap<unsigned int>::get(Ctx));
  auto *Call = IRB.CreateCall(Callee);
  Ret.storeValue(Call);

  return Ret;
}

Var<unsigned int> getGridDimY(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();

  Var<unsigned int> Ret = Fn.declVarInternal<unsigned int>("gridDim.y");

  auto &IRB = Fn.getIRBuilder();
  FunctionCallee Callee = M.getOrInsertFunction(
      "llvm.nvvm.read.ptx.sreg.nctaid.y", TypeMap<unsigned int>::get(Ctx));
  auto *Call = IRB.CreateCall(Callee);
  Ret.storeValue(Call);

  return Ret;
}

Var<unsigned int> getGridDimZ(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();

  Var<unsigned int> Ret = Fn.declVarInternal<unsigned int>("gridDim.z");

  auto &IRB = Fn.getIRBuilder();
  FunctionCallee Callee = M.getOrInsertFunction(
      "llvm.nvvm.read.ptx.sreg.nctaid.z", TypeMap<unsigned int>::get(Ctx));
  auto *Call = IRB.CreateCall(Callee);
  Ret.storeValue(Call);

  return Ret;
}

void syncThreads(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();
  auto &IRB = Fn.getIRBuilder();
  FunctionCallee Callee =
      M.getOrInsertFunction("llvm.nvvm.barrier0", TypeMap<void>::get(Ctx));
  IRB.CreateCall(Callee);
}

} // namespace gpu
} // namespace builtins
} // namespace proteus
