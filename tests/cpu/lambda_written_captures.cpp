// RUN: PROTEUS_AUTO_READONLY_CAPTURES=1 PROTEUS_TRACE_OUTPUT=1 %build/%exe 2>&1 | %FILECHECK %s

#include <iostream>

#include "proteus/JitInterface.h"

int main() {
  proteus::init();

  int A = 10; // Read-only - should be auto-detected
  int B = 20; // Written in lambda - should NOT be auto-detected
  int C = 30; // Read-only - should be auto-detected

  double X[2] = {0.0, 0.0};

  auto lambda = [=, &X]() __attribute__((annotate("jit"))) mutable {
    B = B + 1;
    X[0] = A + B;
    X[1] = C;
  };

  proteus::register_lambda(lambda);
  lambda();

  std::cout << "x[0] = " << X[0] << "\n";
  std::cout << "x[1] = " << X[1] << "\n";

  return 0;
}

// CHECK-DAG: [LambdaSpec][Auto] Replacing slot {{[0-9]+}} with i32 10
// CHECK-DAG: [LambdaSpec][Auto] Replacing slot {{[0-9]+}} with i32 30
// CHECK-NOT: [LambdaSpec][Auto] Replacing slot {{[0-9]+}} with i32 20
// CHECK: x[0] = 31
// CHECK: x[1] = 30
