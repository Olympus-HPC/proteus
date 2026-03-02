// clang-format off
// RUN: %build/mlir_while | %FILECHECK %s
// clang-format on

#include <memory>
#include <proteus/JitFrontend.h>

using namespace proteus;

int main() {
  auto J = std::make_unique<JitModule>("host", "mlir");

  auto &F = J->addFunction<int(int)>("count_down");

  F.beginFunction();
  {
    auto [N] = F.getArgs();
    auto X = F.defVar(N, "x");
    F.beginWhile([&]() { return X > 0; });
    { X -= 1; }
    F.endWhile();
    F.ret(X);
  }
  F.endFunction();

  J->print();

  return 0;
}

// CHECK: scf.while
// CHECK: scf.condition
// CHECK: arith.subi
