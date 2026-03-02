// clang-format off
// RUN: %build/mlir_casts | %FILECHECK %s
// clang-format on

#include <memory>
#include <proteus/Frontend/TypeMap.h>
#include <proteus/JitFrontend.h>

using namespace proteus;

int main() {
  auto J = std::make_unique<JitModule>("host", "mlir");
  auto &F = J->addFunction<void()>("mlir_casts");

  F.beginFunction();
  {
    auto &CB = F.getCodeBuilder();

    IRValue *I32Val = CB.getConstantInt(TypeMap<int>::get(), 7);
    IRValue *BitcastedF32 = CB.createBitCast(I32Val, TypeMap<float>::get());

    IRValue *I1Val = CB.getConstantInt(TypeMap<bool>::get(), 1);
    IRValue *ZExtI32 = CB.createZExt(I1Val, TypeMap<int>::get());

    auto BF = F.declVar<float>("bf");
    auto ZX = F.declVar<int>("zx");
    CB.createStore(BitcastedF32, BF.getSlot());
    CB.createStore(ZExtI32, ZX.getSlot());

    F.ret();
  }
  F.endFunction();

  J->print();
  return 0;
}

// clang-format off
// CHECK: arith.bitcast
// CHECK: arith.extui
// clang-format on
