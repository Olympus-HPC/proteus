// RUN: PROTEUS_AUTO_READONLY_CAPTURES=1 PROTEUS_TRACE_OUTPUT=1 %build/lambda_auto_readonly.%ext 2>&1 | %FILECHECK %s

#include <iostream>

#include "proteus/JitInterface.h"

int main() {
  proteus::init();

  int A = 42;
  double B = 3.14;
  float C = 2.5f;
  bool D = true;

  double X[4] = {0.0, 0.0, 0.0, 0.0};

  auto lambda = [=, &X]() __attribute__((annotate("jit"))) {
    X[0] = A;
    X[1] = B;
    X[2] = C;
    X[3] = D ? 1.0 : 0.0;
  };

  proteus::register_lambda(lambda);
  lambda();

  std::cout << "x[0] = " << X[0] << "\n";
  std::cout << "x[1] = " << X[1] << "\n";
  std::cout << "x[2] = " << X[2] << "\n";
  std::cout << "x[3] = " << X[3] << "\n";

  return 0;
}

// CHECK-DAG: [LambdaSpec][Auto] Replacing slot {{[0-9]+}} with i32 42
// CHECK-DAG: [LambdaSpec][Auto] Replacing slot {{[0-9]+}} with double 3.14
// CHECK-DAG: [LambdaSpec][Auto] Replacing slot {{[0-9]+}} with float 2.5
// CHECK-DAG: [LambdaSpec][Auto] Replacing slot {{[0-9]+}} with i1 1
// CHECK: x[0] = 42
// CHECK: x[1] = 3.14
// CHECK: x[2] = 2.5
// CHECK: x[3] = 1
