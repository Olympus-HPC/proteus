// clang-format off
// RUN: %build/mlir_if | %FILECHECK %s
// clang-format on

#include <memory>
#include <proteus/JitFrontend.h>

using namespace proteus;

int main() {
  auto J = std::make_unique<JitModule>("host", "mlir");

  auto &F = J->addFunction<int(int)>("abs_val");

  F.beginFunction();
  {
    auto [X] = F.getArgs();
    auto Result = F.defVar(X, "result");
    auto IsNeg = X < 0;
    F.beginIf(IsNeg);
    {
      Result = -X;
    }
    F.endIf();
    F.ret(Result);
  }
  F.endFunction();

  J->print();

  return 0;
}

// clang-format off
// CHECK: scf.if
// CHECK: memref.store
