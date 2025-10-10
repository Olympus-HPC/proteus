#include "proteus/Frontend/Builtins.hpp"
#include "proteus/Frontend/Func.hpp"

namespace proteus {
namespace builtins {
namespace gpu {

VarTT<int> getThreadIdX(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();

  VarTT<int> Ret = Fn.declVarTTInternal<int>("threadIdx.x");

  auto &IRB = Fn.getIRBuilder();
  FunctionCallee Callee = M.getOrInsertFunction("llvm.nvvm.read.ptx.sreg.tid.x",
                                                TypeMap<int>::get(Ctx));
  auto *Call = IRB.CreateCall(Callee);
  Ret.Storage->storeValue(Call);

  return Ret;
}

VarTT<int> getBlockIdX(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();

  VarTT<int> Ret = Fn.declVarTTInternal<int>("blockIdx.x");

  auto &IRB = Fn.getIRBuilder();
  FunctionCallee Callee = M.getOrInsertFunction(
      "llvm.nvvm.read.ptx.sreg.ctaid.x", TypeMap<int>::get(Ctx));
  auto *Call = IRB.CreateCall(Callee);
  Ret.Storage->storeValue(Call);

  return Ret;
}

VarTT<int> getBlockDimX(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();

  VarTT<int> Ret = Fn.declVarTTInternal<int>("blockDim.x");

  auto &IRB = Fn.getIRBuilder();
  FunctionCallee Callee = M.getOrInsertFunction(
      "llvm.nvvm.read.ptx.sreg.ntid.x", TypeMap<int>::get(Ctx));
  auto *Call = IRB.CreateCall(Callee);
  Ret.Storage->storeValue(Call);

  return Ret;
}

VarTT<int> getGridDimX(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();

  VarTT<int> Ret = Fn.declVarTTInternal<int>("gridDim.x");

  auto &IRB = Fn.getIRBuilder();
  FunctionCallee Callee = M.getOrInsertFunction(
      "llvm.nvvm.read.ptx.sreg.nctaid.x", TypeMap<int>::get(Ctx));
  auto *Call = IRB.CreateCall(Callee);
  Ret.Storage->storeValue(Call);

  return Ret;
}

VarTT<int> getThreadIdY(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();

  VarTT<int> Ret = Fn.declVarTTInternal<int>("threadIdx.y");

  auto &IRB = Fn.getIRBuilder();
  FunctionCallee Callee = M.getOrInsertFunction("llvm.nvvm.read.ptx.sreg.tid.y",
                                                TypeMap<int>::get(Ctx));
  auto *Call = IRB.CreateCall(Callee);
  Ret.Storage->storeValue(Call);

  return Ret;
}

VarTT<int> getThreadIdZ(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();

  VarTT<int> Ret = Fn.declVarTTInternal<int>("threadIdx.z");

  auto &IRB = Fn.getIRBuilder();
  FunctionCallee Callee = M.getOrInsertFunction("llvm.nvvm.read.ptx.sreg.tid.z",
                                                TypeMap<int>::get(Ctx));
  auto *Call = IRB.CreateCall(Callee);
  Ret.Storage->storeValue(Call);

  return Ret;
}

VarTT<int> getBlockIdY(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();

  VarTT<int> Ret = Fn.declVarTTInternal<int>("blockIdx.y");

  auto &IRB = Fn.getIRBuilder();
  FunctionCallee Callee = M.getOrInsertFunction(
      "llvm.nvvm.read.ptx.sreg.ctaid.y", TypeMap<int>::get(Ctx));
  auto *Call = IRB.CreateCall(Callee);
  Ret.Storage->storeValue(Call);

  return Ret;
}

VarTT<int> getBlockIdZ(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();

  VarTT<int> Ret = Fn.declVarTTInternal<int>("blockIdx.z");

  auto &IRB = Fn.getIRBuilder();
  FunctionCallee Callee = M.getOrInsertFunction(
      "llvm.nvvm.read.ptx.sreg.ctaid.z", TypeMap<int>::get(Ctx));
  auto *Call = IRB.CreateCall(Callee);
  Ret.Storage->storeValue(Call);

  return Ret;
}

VarTT<int> getBlockDimY(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();

  VarTT<int> Ret = Fn.declVarTTInternal<int>("blockDim.y");

  auto &IRB = Fn.getIRBuilder();
  FunctionCallee Callee = M.getOrInsertFunction(
      "llvm.nvvm.read.ptx.sreg.ntid.y", TypeMap<int>::get(Ctx));
  auto *Call = IRB.CreateCall(Callee);
  Ret.Storage->storeValue(Call);

  return Ret;
}

VarTT<int> getBlockDimZ(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();

  VarTT<int> Ret = Fn.declVarTTInternal<int>("blockDim.z");

  auto &IRB = Fn.getIRBuilder();
  FunctionCallee Callee = M.getOrInsertFunction(
      "llvm.nvvm.read.ptx.sreg.ntid.z", TypeMap<int>::get(Ctx));
  auto *Call = IRB.CreateCall(Callee);
  Ret.Storage->storeValue(Call);

  return Ret;
}

VarTT<int> getGridDimY(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();

  VarTT<int> Ret = Fn.declVarTTInternal<int>("gridDim.y");

  auto &IRB = Fn.getIRBuilder();
  FunctionCallee Callee = M.getOrInsertFunction(
      "llvm.nvvm.read.ptx.sreg.nctaid.y", TypeMap<int>::get(Ctx));
  auto *Call = IRB.CreateCall(Callee);
  Ret.Storage->storeValue(Call);

  return Ret;
}

VarTT<int> getGridDimZ(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();

  VarTT<int> Ret = Fn.declVarTTInternal<int>("gridDim.z");

  auto &IRB = Fn.getIRBuilder();
  FunctionCallee Callee = M.getOrInsertFunction(
      "llvm.nvvm.read.ptx.sreg.nctaid.z", TypeMap<int>::get(Ctx));
  auto *Call = IRB.CreateCall(Callee);
  Ret.Storage->storeValue(Call);

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
