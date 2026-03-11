// clang-format off
// RUN: %build/mlir_intrinsics | %FILECHECK %s
// clang-format on

#include <memory>
#include <proteus/JitFrontend.h>

using namespace proteus;

int main() {
  auto J = std::make_unique<JitModule>("host", "mlir");
  auto &F = J->addFunction<float(float)>("intrinsics");

  F.beginFunction();
  {
    auto [X] = F.getArgs();
    auto S = sinf(X);
    auto C = cosf(S);
    auto E = expf(C);
    auto L = logf(E);
    auto Q = sqrtf(L);
    auto T = truncf(Q);
    auto A = absf(T);
    F.ret(A);
  }
  F.endFunction();

  J->print();
  return 0;
}

// clang-format off
// CHECK: func.func @intrinsics
// CHECK: math.sin
// CHECK: math.cos
// CHECK: math.exp
// CHECK: math.log
// CHECK: math.sqrt
// CHECK: math.trunc
// CHECK: arith.negf
// CHECK: arith.maximumf
// clang-format on
