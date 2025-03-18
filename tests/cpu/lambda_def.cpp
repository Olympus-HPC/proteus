// RUN: rm -rf .proteus
// RUN: ./lambda_def | FileCheck %s --check-prefixes=CHECK
// RUN: rm -rf .proteus

#include <cstdio>

#include <proteus/JitInterface.hpp>

template <typename F> void run(F &&Func) {
  proteus::register_lambda(Func);
  Func();
}

int main(int argc, char **argv) {
  proteus::init();

  int A = 42;
  auto Lambda =
      [ =, A = proteus::jit_variable(A) ]() __attribute__((annotate("jit"))) {
    printf("Lambda A %d\n", A);
  };
  run(Lambda);
  run(Lambda);
  run(Lambda);

  proteus::finalize();
  return 0;
}

// CHECK-3: Lambda 42
// CHECK: JitCache hits 2 total 3
// CHECK: HashValue {{[0-9]+}} NumExecs 3 NumHits 2
