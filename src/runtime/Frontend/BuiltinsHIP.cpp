#include "proteus/Frontend/Builtins.h"
#include "proteus/Frontend/Func.h"

#include <llvm/IR/Module.h>

namespace proteus {
namespace builtins {
namespace gpu {

namespace detail {

// (no llvm:: namespace needed here)

// Offsets in implicit arg pts in i32 step.
constexpr unsigned OffsetGridDimX = 0;
constexpr unsigned OffsetGridDimY = 1;
constexpr unsigned OffsetGridDimZ = 2;

static inline IRValue getGridDim(FuncBase &Fn, unsigned Offset) {
  // An alternative way is by using __ockl_get_num_groups but needs to link with
  // hip bitcode libraries.
  // Address space 4 is the AMDGPU constant/readonly address space used by
  // the implicit kernel argument pointer ABI.
  constexpr unsigned ConstantAddressSpace = 4;
  auto Call = Fn.getCodeBuilder().createCall("llvm.amdgcn.implicitarg.ptr",
                                             IRType{IRTypeKind::Pointer, false,
                                                    0, IRTypeKind::Void,
                                                    ConstantAddressSpace});
  auto GEP = Fn.getCodeBuilder().createInBoundsGEP(
      IRType{IRTypeKind::Int32}, Call,
      {Fn.getCodeBuilder().getConstantInt(IRType{IRTypeKind::Int64}, Offset)});
  auto Load = Fn.getCodeBuilder().createLoad(IRType{IRTypeKind::Int32}, GEP);

  return Load;
}

// Offsets in implicit arg pts in i16 step.
constexpr unsigned OffsetBlockDimX = 6;
constexpr unsigned OffsetBlockDimY = 7;
constexpr unsigned OffsetBlockDimZ = 8;

static inline IRValue getBlockDim(FuncBase &Fn, unsigned Offset) {
  // An alternative way is by using __ockl_get_local_size but needs to link with
  // hip bitcode libraries.
  // Address space 4 is the AMDGPU constant/readonly address space used by
  // the implicit kernel argument pointer ABI.
  constexpr unsigned ConstantAddressSpace = 4;
  auto Call = Fn.getCodeBuilder().createCall("llvm.amdgcn.implicitarg.ptr",
                                             IRType{IRTypeKind::Pointer, false,
                                                    0, IRTypeKind::Void,
                                                    ConstantAddressSpace});
  auto GEP = Fn.getCodeBuilder().createInBoundsGEP(
      IRType{IRTypeKind::Int16}, Call,
      {Fn.getCodeBuilder().getConstantInt(IRType{IRTypeKind::Int64}, Offset)});
  auto Load = Fn.getCodeBuilder().createLoad(IRType{IRTypeKind::Int16}, GEP);
  auto Conv = Fn.getCodeBuilder().createZExt(Load, IRType{IRTypeKind::Int32});

  return Conv;
}

} // namespace detail

Var<unsigned int> getThreadIdX(FuncBase &Fn) {

  Var<unsigned int> Ret = Fn.declVar<unsigned int>("threadIdx.x");

  auto Call = Fn.getCodeBuilder().createCall("llvm.amdgcn.workitem.id.x",
                                             TypeMap<unsigned int>::get());
  Ret.storeValue(Call);

  return Ret;
}

Var<unsigned int> getBlockIdX(FuncBase &Fn) {

  Var<unsigned int> Ret = Fn.declVar<unsigned int>("blockIdx.x");

  auto Call = Fn.getCodeBuilder().createCall("llvm.amdgcn.workgroup.id.x",
                                             TypeMap<unsigned int>::get());
  Ret.storeValue(Call);

  return Ret;
}

Var<unsigned int> getBlockDimX(FuncBase &Fn) {
  Var<unsigned int> Ret = Fn.declVar<unsigned int>("blockDim.x");

  IRValue Conv = detail::getBlockDim(Fn, detail::OffsetBlockDimX);
  Ret.storeValue(Conv);

  return Ret;
}

Var<unsigned int> getGridDimX(FuncBase &Fn) {
  Var<unsigned int> Ret = Fn.declVar<unsigned int>("gridDim.x");

  IRValue Conv = detail::getGridDim(Fn, detail::OffsetGridDimX);
  Ret.storeValue(Conv);

  return Ret;
}

Var<unsigned int> getThreadIdY(FuncBase &Fn) {

  Var<unsigned int> Ret = Fn.declVar<unsigned int>("threadIdx.y");

  auto Call = Fn.getCodeBuilder().createCall("llvm.amdgcn.workitem.id.y",
                                             TypeMap<unsigned int>::get());
  Ret.storeValue(Call);

  return Ret;
}

Var<unsigned int> getThreadIdZ(FuncBase &Fn) {

  Var<unsigned int> Ret = Fn.declVar<unsigned int>("threadIdx.z");

  auto Call = Fn.getCodeBuilder().createCall("llvm.amdgcn.workitem.id.z",
                                             TypeMap<unsigned int>::get());
  Ret.storeValue(Call);

  return Ret;
}

Var<unsigned int> getBlockIdY(FuncBase &Fn) {

  Var<unsigned int> Ret = Fn.declVar<unsigned int>("blockIdx.y");

  auto Call = Fn.getCodeBuilder().createCall("llvm.amdgcn.workgroup.id.y",
                                             TypeMap<unsigned int>::get());
  Ret.storeValue(Call);

  return Ret;
}

Var<unsigned int> getBlockIdZ(FuncBase &Fn) {

  Var<unsigned int> Ret = Fn.declVar<unsigned int>("blockIdx.z");

  auto Call = Fn.getCodeBuilder().createCall("llvm.amdgcn.workgroup.id.z",
                                             TypeMap<unsigned int>::get());
  Ret.storeValue(Call);

  return Ret;
}

Var<unsigned int> getBlockDimY(FuncBase &Fn) {
  Var<unsigned int> Ret = Fn.declVar<unsigned int>("blockDim.y");

  IRValue Conv = detail::getBlockDim(Fn, detail::OffsetBlockDimY);
  Ret.storeValue(Conv);

  return Ret;
}

Var<unsigned int> getBlockDimZ(FuncBase &Fn) {
  Var<unsigned int> Ret = Fn.declVar<unsigned int>("blockDim.z");

  IRValue Conv = detail::getBlockDim(Fn, detail::OffsetBlockDimZ);
  Ret.storeValue(Conv);

  return Ret;
}

Var<unsigned int> getGridDimY(FuncBase &Fn) {
  Var<unsigned int> Ret = Fn.declVar<unsigned int>("gridDim.y");

  IRValue Conv = detail::getGridDim(Fn, detail::OffsetGridDimY);
  Ret.storeValue(Conv);

  return Ret;
}

Var<unsigned int> getGridDimZ(FuncBase &Fn) {
  Var<unsigned int> Ret = Fn.declVar<unsigned int>("gridDim.z");

  IRValue Conv = detail::getGridDim(Fn, detail::OffsetGridDimZ);
  Ret.storeValue(Conv);

  return Ret;
}

void syncThreads(FuncBase &Fn) {
  Fn.getCodeBuilder().createCall("llvm.amdgcn.s.barrier", TypeMap<void>::get());
}

} // namespace gpu
} // namespace builtins
} // namespace proteus
