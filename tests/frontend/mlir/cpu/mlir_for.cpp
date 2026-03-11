// clang-format off
// RUN: %build/mlir_for | %FILECHECK %s
// clang-format on

#include <memory>
#include <proteus/JitFrontend.h>

using namespace proteus;

int main() {
  auto J = std::make_unique<JitModule>("host", "mlir");

  auto &F = J->addFunction<int(int)>("sum_to_n");

  F.beginFunction();
  {
    auto [N] = F.getArgs();
    auto Sum = F.declVar<int>("sum");
    auto I = F.declVar<int>("i");
    Sum = 0;
    auto Zero = F.defVar<int>(0);
    auto One = F.defVar<int>(1);
    F.beginFor(I, Zero, N, One);
    {
      Sum += I;
    }
    F.endFor();
    F.ret(Sum);
  }
  F.endFunction();

  J->print();

  return 0;
}

// clang-format off
// CHECK: scf.for
// CHECK: arith.index_cast
// CHECK: memref.store
// CHECK: arith.addi
