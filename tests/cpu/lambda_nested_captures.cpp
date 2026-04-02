// clang-format off
// RUN: PROTEUS_AUTO_READONLY_CAPTURES=1 PROTEUS_TRACE_OUTPUT=specialization %build/%exe lambda_nested_captures 2>&1 | %FILECHECK %s
// clang-format on

#include <iostream>

#include "proteus/JitInterface.h"

struct Payload {
  int A;
  double B;
};

int main() {
  Payload P{42, 3.14};
  double X[2] = {0.0, 0.0};

  auto lambda = [=, &X]() __attribute__((annotate("jit"))) {
    X[0] = P.A;
    X[1] = P.B;
  };

  proteus::register_lambda(lambda);
  lambda();

  std::cout << "x[0] = " << X[0] << "\n";
  std::cout << "x[1] = " << X[1] << "\n";

  return 0;
}

// CHECK-DAG: [LambdaSpec][Auto] Replacing slot {{[0-9]+}} with i32 42
// CHECK-DAG: [LambdaSpec][Auto] Replacing slot {{[0-9]+}} with double 3.14
// CHECK: x[0] = 42
// CHECK: x[1] = 3.14
