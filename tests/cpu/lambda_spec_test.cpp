// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT=1 %build/lambda_def | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/lambda_def | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>

#include <proteus/JitInterface.h>

template<typename Lambda>
class Abstraction {
  public:
  Lambda lambda;
  Abstraction(Lambda&& lam) : lambda(lam) {};
  auto operator()() {
    return lambda();
  }
};

void printInt(int I) { printf("Integer = %d\n", I); }

template <typename T> void run(T &&LB) {
  proteus::register_lambda(LB);
  Abstraction<T> abs (LB) ;
  run2(abs);
}

template<typename T> void run2(T&& LB) {
  LB();
}

int main() {
  proteus::init();

  int zero = 0;
  int one = 1;
  int two = 2;

  auto zero_lambda = [=, c = proteus::jit_variable(zero)] ()
                         __attribute__((annotate("jit"))) { printInt(c); };

  auto one_lambda = [=, c = proteus::jit_variable(one)] ()
                        __attribute__((annotate("jit"))) { printInt(c); };

  auto two_lambda = [=, c = proteus::jit_variable(two)] ()
                        __attribute__((annotate("jit"))) { printInt(c); };

  run(zero_lambda);
  run(one_lambda);
  run(two_lambda);

  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK-FIRST: [LambdaSpec] Function: _ZZ4mainENKUlvE_clEv RCVec size 1
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 0
// CHECK Integer = 0
// CHECK-FIRST: [LambdaSpec] Function: _ZZ4mainENKUlvE0_clEv RCVec size 1
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 1
// CHECK Integer = 1
// CHECK-FIRST: [LambdaSpec] Function: _ZZ4mainENKUlvE1_clEv RCVec size 1
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 2
// CHECK Integer = 2
