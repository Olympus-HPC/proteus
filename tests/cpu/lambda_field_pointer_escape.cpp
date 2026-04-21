// clang-format off
// RUN: PROTEUS_AUTO_READONLY_CAPTURES=1 PROTEUS_TRACE_OUTPUT=specialization %build/%exe lambda_field_pointer_escape 2>&1 | %FILECHECK %s
// clang-format on

#include <cstdlib>
#include <iostream>

#include "proteus/JitInterface.h"

__attribute__((noinline)) void observe(const int *Ptr) {
  if (!Ptr)
    std::abort();
}

int main() {
  int A = 42;
  double B = 3.14;
  double X[2] = {0.0, 0.0};

  auto lambda = [&X, A, B]() __attribute__((annotate("jit"))) {
    observe(&A);
    X[0] = A;
    X[1] = B;
  };

  proteus::register_lambda(lambda);
  lambda();

  std::cout << "x[0] = " << X[0] << "\n";
  std::cout << "x[1] = " << X[1] << "\n";

  return 0;
}

// CHECK-DAG: [LambdaSpec][Auto] Replacing slot {{[0-9]+}} with double 3.14
// CHECK-NOT: [LambdaSpec][Auto] Replacing slot {{[0-9]+}} with i32 42
// CHECK: x[0] = 42
// CHECK: x[1] = 3.14
