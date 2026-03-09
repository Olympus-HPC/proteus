// clang-format off
// RUN: %build/mlir_add_vectors | %FILECHECK %s
// clang-format on

#include <memory>
#include <proteus/JitFrontend.h>

using namespace proteus;

int main() {
  auto J = std::make_unique<JitModule>("host", "mlir");

  auto &F = J->addFunction<void(double *, double *, int)>("add_vectors");

  F.beginFunction();
  {
    auto [A, B, N] = F.getArgs();
    auto I = F.declVar<int>("I");
    auto Zero = F.defVar<int>(0);
    auto One = F.defVar<int>(1);
    F.beginFor(I, Zero, N, One);
    {
      A[I] = A[I] + B[I];
    }
    F.endFor();
    F.ret();
  }
  F.endFunction();

  J->print();

  return 0;
}

// clang-format off
// CHECK: func.func @add_vectors
// CHECK: scf.for
// CHECK: memref.load
// CHECK: arith.addf
// CHECK: memref.store
