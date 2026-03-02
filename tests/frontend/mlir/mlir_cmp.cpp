// clang-format off
// RUN: %build/mlir_cmp | %FILECHECK %s
// clang-format on

#include <memory>
#include <proteus/JitFrontend.h>

using namespace proteus;

int main() {
  auto J = std::make_unique<JitModule>("host", "mlir");

  auto &F = J->addFunction<int(int, int)>("max_val");

  F.beginFunction();
  {
    auto [A, B] = F.getArgs();
    auto Result = F.defVar(A, "result");
    auto Cond = B > A;
    F.beginIf(Cond);
    { Result = B; }
    F.endIf();
    F.ret(Result);
  }
  F.endFunction();

  J->print();

  return 0;
}

// CHECK: arith.cmpi sgt
// CHECK: scf.if
// CHECK: memref.store
