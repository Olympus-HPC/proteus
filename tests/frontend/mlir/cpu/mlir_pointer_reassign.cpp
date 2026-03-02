// clang-format off
// RUN: %build/mlir_pointer_reassign | %FILECHECK %s
// clang-format on

#include <memory>
#include <proteus/JitFrontend.h>

using namespace proteus;

int main() {
  auto J = std::make_unique<JitModule>("host", "mlir");

  auto &F = J->addFunction<void()>("foo");

  F.beginFunction();
  {
    auto A = F.declVar<int[]>(4, AddressSpace::DEFAULT, "A");
    auto B = F.declVar<int[]>(4, AddressSpace::DEFAULT, "B");
    auto P = F.declVar<int *>("p");

    P.storeAddress(A[0].getAddress().loadAddress());
    P.storeAddress(B[0].getAddress().loadAddress());

    (*P) = 7;

    F.ret();
  }
  F.endFunction();

  J->print();
  return 0;
}

// clang-format off
// CHECK-LABEL: func.func @foo
// CHECK: llvm.store %c7_i32, %{{.*}} : i32, !llvm.ptr
// clang-format on
