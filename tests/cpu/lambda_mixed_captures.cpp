// RUN: PROTEUS_AUTO_READONLY_CAPTURES=0 PROTEUS_TRACE_OUTPUT=1 %build/%exe 2>&1 | %FILECHECK %s

#include <iostream>

#include "proteus/JitInterface.h"

int main() {
  proteus::init();

  int A = 10;
  int B = 20;
  double C = 3.14;
  double D = 2.71;

  double X[2] = {0.0, 0.0};

  auto lambda = [=, &X,
                 A = proteus::jit_variable(A),
                 C = proteus::jit_variable(C)] __attribute__((annotate("jit"))) {
    X[0] = A + B;
    X[1] = C + D;
  };

  proteus::register_lambda(lambda);
  lambda();

  std::cout << "x[0] = " << X[0] << "\n";
  std::cout << "x[1] = " << X[1] << "\n";

  return 0;
}

// CHECK-DAG: [LambdaSpec] Replacing slot {{[0-9]+}} with i32 10
// CHECK-DAG: [LambdaSpec] Replacing slot {{[0-9]+}} with double 3.14
// CHECK: x[0] = 30
// CHECK: x[1] = 5.85
