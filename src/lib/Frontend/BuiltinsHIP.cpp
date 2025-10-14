#include "proteus/Frontend/Builtins.hpp"
#include "proteus/Frontend/Func.hpp"

namespace proteus {
namespace builtins {
namespace gpu {

namespace detail {

// Offsets in implicit arg pts in i32 step.
constexpr unsigned OffsetGridDimX = 0;
constexpr unsigned OffsetGridDimY = 1;
constexpr unsigned OffsetGridDimZ = 2;

static inline Value *getGridDim(FuncBase &Fn, unsigned Offset) {
  // An alternative way is by using __ockl_get_num_groups but needs to link with
  // hip bitcode libraries.
  constexpr int ConstantAddressSpace = 4;
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();

  auto &IRB = Fn.getIRBuilder();
  FunctionCallee Callee =
      M.getOrInsertFunction("llvm.amdgcn.implicitarg.ptr",
                            PointerType::get(Ctx, ConstantAddressSpace));
  auto *Call = IRB.CreateCall(Callee);
  auto *GEP = IRB.CreateInBoundsGEP(
      IRB.getInt32Ty(), Call, {ConstantInt::get(IRB.getInt64Ty(), Offset)});
  auto *Load = IRB.CreateLoad(IRB.getInt32Ty(), GEP);

  return Load;
}

// Offsets in implicit arg pts in i16 step.
constexpr unsigned OffsetBlockDimX = 6;
constexpr unsigned OffsetBlockDimY = 7;
constexpr unsigned OffsetBlockDimZ = 8;

static inline Value *getBlockDim(FuncBase &Fn, unsigned Offset) {
  // An alternative way is by using __ockl_get_local_size but needs to link with
  // hip bitcode libraries.
  constexpr int ConstantAddressSpace = 4;
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();

  auto &IRB = Fn.getIRBuilder();
  FunctionCallee Callee =
      M.getOrInsertFunction("llvm.amdgcn.implicitarg.ptr",
                            PointerType::get(Ctx, ConstantAddressSpace));
  auto *Call = IRB.CreateCall(Callee);
  auto *GEP = IRB.CreateInBoundsGEP(
      IRB.getInt16Ty(), Call, {ConstantInt::get(IRB.getInt64Ty(), Offset)});
  auto *Load = IRB.CreateLoad(IRB.getInt16Ty(), GEP);
  auto *Conv = IRB.CreateZExt(Load, IRB.getInt32Ty());

  return Conv;
}

} // namespace detail

Var<int> getThreadIdX(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();

  Var<int> Ret = Fn.declVarInternal<int>("threadIdx.x");

  auto &IRB = Fn.getIRBuilder();
  FunctionCallee Callee = M.getOrInsertFunction("llvm.amdgcn.workitem.id.x",
                                                TypeMap<int>::get(Ctx));
  auto *Call = IRB.CreateCall(Callee);
  Ret.storeValue(Call);

  return Ret;
}

Var<int> getBlockIdX(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();

  Var<int> Ret = Fn.declVarInternal<int>("blockIdx.x");

  auto &IRB = Fn.getIRBuilder();
  FunctionCallee Callee = M.getOrInsertFunction("llvm.amdgcn.workgroup.id.x",
                                                TypeMap<int>::get(Ctx));
  auto *Call = IRB.CreateCall(Callee);
  Ret.storeValue(Call);

  return Ret;
}

Var<int> getBlockDimX(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  Var<int> Ret = Fn.declVarInternal<int>("blockDim.x");

  Value *Conv = detail::getBlockDim(Fn, detail::OffsetBlockDimX);
  Ret.storeValue(Conv);

  return Ret;
}

Var<int> getGridDimX(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  Var<int> Ret = Fn.declVarInternal<int>("gridDim.x");

  Value *Conv = detail::getGridDim(Fn, detail::OffsetGridDimX);
  Ret.storeValue(Conv);

  return Ret;
}

Var<int> getThreadIdY(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();

  Var<int> Ret = Fn.declVarInternal<int>("threadIdx.y");

  auto &IRB = Fn.getIRBuilder();
  FunctionCallee Callee = M.getOrInsertFunction("llvm.amdgcn.workitem.id.y",
                                                TypeMap<int>::get(Ctx));
  auto *Call = IRB.CreateCall(Callee);
  Ret.storeValue(Call);

  return Ret;
}

Var<int> getThreadIdZ(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();

  Var<int> Ret = Fn.declVarInternal<int>("threadIdx.z");

  auto &IRB = Fn.getIRBuilder();
  FunctionCallee Callee = M.getOrInsertFunction("llvm.amdgcn.workitem.id.z",
                                                TypeMap<int>::get(Ctx));
  auto *Call = IRB.CreateCall(Callee);
  Ret.storeValue(Call);

  return Ret;
}

Var<int> getBlockIdY(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();

  Var<int> Ret = Fn.declVarInternal<int>("blockIdx.y");

  auto &IRB = Fn.getIRBuilder();
  FunctionCallee Callee = M.getOrInsertFunction("llvm.amdgcn.workgroup.id.y",
                                                TypeMap<int>::get(Ctx));
  auto *Call = IRB.CreateCall(Callee);
  Ret.storeValue(Call);

  return Ret;
}

Var<int> getBlockIdZ(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();

  Var<int> Ret = Fn.declVarInternal<int>("blockIdx.z");

  auto &IRB = Fn.getIRBuilder();
  FunctionCallee Callee = M.getOrInsertFunction("llvm.amdgcn.workgroup.id.z",
                                                TypeMap<int>::get(Ctx));
  auto *Call = IRB.CreateCall(Callee);
  Ret.storeValue(Call);

  return Ret;
}

Var<int> getBlockDimY(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  Var<int> Ret = Fn.declVarInternal<int>("blockDim.y");

  Value *Conv = detail::getBlockDim(Fn, detail::OffsetBlockDimY);
  Ret.storeValue(Conv);

  return Ret;
}

Var<int> getBlockDimZ(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  Var<int> Ret = Fn.declVarInternal<int>("blockDim.z");

  Value *Conv = detail::getBlockDim(Fn, detail::OffsetBlockDimZ);
  Ret.storeValue(Conv);

  return Ret;
}

Var<int> getGridDimY(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  Var<int> Ret = Fn.declVarInternal<int>("gridDim.y");

  Value *Conv = detail::getGridDim(Fn, detail::OffsetGridDimY);
  Ret.storeValue(Conv);

  return Ret;
}

Var<int> getGridDimZ(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  Var<int> Ret = Fn.declVarInternal<int>("gridDim.z");

  Value *Conv = detail::getGridDim(Fn, detail::OffsetGridDimZ);
  Ret.storeValue(Conv);

  return Ret;
}

void syncThreads(FuncBase &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();
  auto &IRB = Fn.getIRBuilder();
  FunctionCallee Callee =
      M.getOrInsertFunction("llvm.amdgcn.s.barrier", TypeMap<void>::get(Ctx));
  IRB.CreateCall(Callee);
}

} // namespace gpu
} // namespace builtins
} // namespace proteus
