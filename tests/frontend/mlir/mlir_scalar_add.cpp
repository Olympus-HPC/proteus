// clang-format off
// RUN: %build/mlir_scalar_add | %FILECHECK %s
// clang-format on

#include <memory>
#include <proteus/JitFrontend.h>

using namespace proteus;

int main() {
  // Construct a JitModule targeting the host with the MLIR backend.
  auto J = std::make_unique<JitModule>("host", "mlir");

  // Declare a function:  int add(int A, int B)
  auto &F = J->addFunction<int(int, int)>("add");

  F.beginFunction();
  {
    // Retrieve typed arguments.
    auto [A, B] = F.getArgs();
    // Compute C = A + B.
    auto C = A + B;
    // Return C.
    F.ret(C);
  }
  F.endFunction();

  // Print the MLIR module to stdout. FileCheck verifies the output.
  J->print();

  return 0;
}

// CHECK: module
// CHECK: func.func @add
// CHECK: memref.alloca
// CHECK: arith.addi
// CHECK: return
