// RUN: PROTEUS_AUTO_READONLY_CAPTURES=0 PROTEUS_TRACE_OUTPUT=1 %build/%exe 2>&1 | %FILECHECK %s

#include <iostream>

#include "proteus/JitInterface.h"

int main() {
  proteus::init();

  int Scalar = 42;
  int *Ptr = &Scalar;
  double Value = 3.14;

  double X[2] = {0.0, 0.0};

  auto lambda = [=, &X]() __attribute__((annotate("jit"))) {
    X[0] = Scalar;
    X[1] = Value;
    (void)Ptr;
  };

  int *PtrOnly = &Scalar;
  auto lambda2 = [=]() __attribute__((annotate("jit"))) { (void)PtrOnly; };

  proteus::register_lambda(lambda);
  proteus::register_lambda(lambda2);

  lambda();
  lambda2();

  std::cout << "x[0] = " << X[0] << "\n";
  std::cout << "x[1] = " << X[1] << "\n";

  return 0;
}

// CHECK: x[0] = 42
// CHECK: x[1] = 3.14
