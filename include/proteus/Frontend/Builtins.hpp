#ifndef PROTEUS_FRONTEND_BUILTINS_HPP
#define PROTEUS_FRONTEND_BUILTINS_HPP

#include "proteus/Frontend/Func.hpp"

namespace proteus {
namespace builtins {

#if PROTEUS_ENABLE_HIP
namespace hip {

namespace detail {

// Offsets in implicit arg pts in i32 step.
constexpr unsigned OffsetGridDimX = 0;
constexpr unsigned OffsetGridDimY = 1;
constexpr unsigned OffsetGridDimZ = 2;

inline Value *getGridDim(Func &Fn, unsigned Offset) {
  // An alternative way is by using __ockl_get_num_groups but needs to link with
  // hip bitcode libraries.
  constexpr int ConstantAddressSpace = 4;
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();

  auto &IRB = Fn.getIRB();
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

inline Value *getBlockDim(Func &Fn, unsigned Offset) {
  // An alternative way is by using __ockl_get_local_size but needs to link with
  // hip bitcode libraries.
  constexpr int ConstantAddressSpace = 4;
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();

  auto &IRB = Fn.getIRB();
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

inline Var &getThreadIdX(Func &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();

  Var &Ret = Fn.declVarInternal("threadIdx.x", TypeMap<int>::get(Ctx));

  auto &IRB = Fn.getIRB();
  FunctionCallee Callee = M.getOrInsertFunction("llvm.amdgcn.workitem.id.x",
                                                TypeMap<int>::get(Ctx));
  auto *Call = IRB.CreateCall(Callee);
  Ret.storeValue(Call);

  return Ret;
}

inline Var &getBlockIdX(Func &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  auto &M = *Fn.getFunction()->getParent();

  Var &Ret = Fn.declVarInternal("blockIdx.x", TypeMap<int>::get(Ctx));

  auto &IRB = Fn.getIRB();
  FunctionCallee Callee = M.getOrInsertFunction("llvm.amdgcn.workgroup.id.x",
                                                TypeMap<int>::get(Ctx));
  auto *Call = IRB.CreateCall(Callee);
  Ret.storeValue(Call);

  return Ret;
}

inline Var &getBlockDimX(Func &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  // TODO: Return an "int" variable, could be a different type.
  Var &Ret = Fn.declVarInternal("blockDim.x", TypeMap<int>::get(Ctx));

  Value *Conv = detail::getBlockDim(Fn, detail::OffsetBlockDimX);
  Ret.storeValue(Conv);

  return Ret;
}

inline Var &getGridDimX(Func &Fn) {
  auto &Ctx = Fn.getFunction()->getContext();
  Var &Ret = Fn.declVarInternal("gridDim.x", TypeMap<int>::get(Ctx));

  Value *Conv = detail::getGridDim(Fn, detail::OffsetGridDimX);
  Ret.storeValue(Conv);

  return Ret;
}

} // namespace hip
#endif

#if PROTEUS_ENABLE_CUDA
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
#endif
} // namespace builtins
} // namespace proteus

#endif