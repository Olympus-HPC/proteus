// clang-format off
// RUN: %build/mlir_array | %FILECHECK %s
// clang-format on

#include <memory>
#include <proteus/JitFrontend.h>

using namespace proteus;

int main() {
  auto J = std::make_unique<JitModule>("host", "mlir");

  auto &F = J->addFunction<void(double *)>("array_test");

  F.beginFunction();
  {
    auto [Out] = F.getArgs();
    auto Arr = F.declVar<double[]>(4, AddressSpace::DEFAULT, "arr");
    auto I = F.declVar<int>("I");
    auto Zero = F.defVar<int>(0);
    auto Four = F.defVar<int>(4);
    auto One = F.defVar<int>(1);

    // Fill Arr[I] = 1.0 * I.
    F.beginFor(I, Zero, Four, One);
    { Arr[I] = 1.0 * I; }
    F.endFor();

    // Copy Arr[I] to Out[I].
    F.beginFor(I, Zero, Four, One);
    { Out[I] = Arr[I]; }
    F.endFor();

    F.ret();
  }
  F.endFunction();

  J->print();

  return 0;
}

// clang-format off
// CHECK: memref.alloca() : memref<4xf64>
// CHECK: scf.for
// CHECK: memref.store
// CHECK: memref.load
