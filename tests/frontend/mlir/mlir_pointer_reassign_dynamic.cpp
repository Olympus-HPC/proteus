// clang-format off
// RUN: %build/mlir_pointer_reassign_dynamic | %FILECHECK %s
// clang-format on

#include <memory>
#include <proteus/JitFrontend.h>

using namespace proteus;

int main() {
  auto J = std::make_unique<JitModule>("host", "mlir");

  auto &F = J->addFunction<void(int)>("foo");

  F.beginFunction();
  {
    auto [I] = F.getArgs();
    auto A = F.declVar<int[]>(8, AddressSpace::DEFAULT, "A");
    auto B = F.declVar<int[]>(8, AddressSpace::DEFAULT, "B");
    auto P = F.declVar<int *>("p");

    P.storeAddress(A[0].getAddress().loadAddress());
    P.storeAddress(B[I].getAddress().loadAddress());

    (*P) = 1;

    F.ret();
  }
  F.endFunction();

  J->print();
  return 0;
}

// clang-format off
// CHECK-LABEL: func.func @foo
// CHECK: %[[IDX0:.*]] = arith.index_cast %{{.*}} : i32 to index
// CHECK: memref.store %[[IDX0]], %{{.*}}[%{{.*}}] : memref<1xindex>
// CHECK: memref.store %c1_i32, %{{.*}}[%{{.*}}] : memref<8xi32>
// clang-format on
