// clang-format off
// RUN: %build/mlir_generic_load_store | %FILECHECK %s
// clang-format on

#include <memory>
#include <proteus/Frontend/TypeMap.h>
#include <proteus/JitFrontend.h>

using namespace proteus;

int main() {
  auto J = std::make_unique<JitModule>("host", "mlir");
  auto &F = J->addFunction<void()>("generic_load_store");

  F.beginFunction();
  {
    auto &CB = F.getCodeBuilder();

    // --- Scalar-slot load/store path ---
    auto X = F.declVar<int>("x");
    auto Y = F.declVar<int>("y");

    IRValue *C11 = CB.getConstantInt(TypeMap<int>::get(), 11);
    CB.createStore(C11, X.getSlot());

    IRValue *LX = CB.createLoad(TypeMap<int>::get(), X.getSlot(), "lx");
    CB.createStore(LX, Y.getSlot());

    // --- Pointer-dereference load/store path ---
    auto A = F.declVar<int[]>(4, AddressSpace::DEFAULT, "A");
    auto P = F.declVar<int *>("p");
    auto Z = F.declVar<int>("z");

    // p = &A[0]
    P.storeAddress(A[0].getAddress().loadAddress());

    // *p = 42
    IRValue *C42 = CB.getConstantInt(TypeMap<int>::get(), 42);
    CB.createStore(C42, P.loadAddress());

    // z = *p
    IRValue *LP = CB.createLoad(TypeMap<int>::get(), P.loadAddress(), "lp");
    CB.createStore(LP, Z.getSlot());

    F.ret();
  }
  F.endFunction();

  J->print();
  return 0;
}

// clang-format off
// CHECK-LABEL: func.func @generic_load_store

// Scalar slot operations: memref<1xi32> at [0].
// CHECK: memref.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<1xi32>
// CHECK: %{{.*}} = memref.load %{{.*}}[%{{.*}}] : memref<1xi32>

// Pointer dereference operations: index comes from loading pointer slot.
// CHECK: %c42_i32 = arith.constant 42 : i32
// CHECK: %{{.*}} = memref.load %{{.*}}[%{{.*}}] : memref<1x!llvm.ptr>
// CHECK: llvm.store %c42_i32, %{{.*}} : i32, !llvm.ptr
// CHECK: %{{.*}} = memref.load %{{.*}}[%{{.*}}] : memref<1x!llvm.ptr>
// CHECK: %{{.*}} = llvm.load %{{.*}} : !llvm.ptr -> i32
// clang-format on
