// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization;cache-stats" %build/lambda_factory | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="cache-stats" %build/lambda_factory | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>
#include <utility>

#include <proteus/JitInterface.h>

void printInt(int I) { printf("Integer = %d\n", I); }

auto declareLambda(int rc1, int rc2) {
  return proteus::register_lambda(
      [=, C = proteus::jit_variable(rc1), D = 5,
       C2 = proteus::jit_variable(rc2)]() __attribute__((annotate("jit"))) {
        printInt(C);
        printInt(C2);
        printInt(D);
      });
}

template <typename L> struct Abstraction {
  L lambda_1;
  L lambda_2;
  L lambda_3;

  Abstraction(L &&l1, L &&l2, L &&l3)
      : lambda_1(std::move(l1)), lambda_2(std::move(l2)), lambda_3(std::move(l3)) {}

  auto operator()() {
    lambda_1();
    lambda_2();
    lambda_3();
  }
};

auto declareLambdaThree(int rc1, int rc2, int rc3) {
  return [=, C = proteus::jit_variable(rc1), D = 4,
          C2 = proteus::jit_variable(rc2), rc3 = proteus::jit_variable(rc3)]()
             __attribute__((annotate("jit"))) {
    printInt(C);
    printInt(C2);
    printInt(rc3);
    printInt(D);
  };
}

auto forwardLambda(int rc1, int, int rc3) {
  int loc = 1;
  return declareLambdaThree(rc1, loc, rc3);
}

int main() {
  int Zero = 0;
  int One = 1;
  int Two = 2;

  // `declareLambda` registers its lambda at a fixed source location, so all
  // calls share a single FunctorID but with different runtime values.
  auto ZeroLambda = declareLambda(Zero, One);
  auto OneLambda = declareLambda(One, Two);
  auto TwoLambda = declareLambda(Two, Zero);

  Abstraction A(std::move(ZeroLambda), std::move(OneLambda),
                std::move(TwoLambda));

  auto BigLam = proteus::register_lambda(declareLambdaThree(Zero, One, Two));
  auto ForLam = proteus::register_lambda(forwardLambda(Zero, One, Two));

  A();
  BigLam();
  ForLam();

  return 0;
}

// clang-format off
// CHECK-FIRST: [LambdaSpec] Replacing slot 2 with i32 1
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 0
// CHECK: Integer = 0
// CHECK: Integer = 1
// CHECK: Integer = 5
// CHECK-FIRST: [LambdaSpec] Replacing slot 2 with i32 2
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 1
// CHECK: Integer = 1
// CHECK: Integer = 2
// CHECK: Integer = 5
// CHECK-FIRST: [LambdaSpec] Replacing slot 2 with i32 0
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 2
// CHECK: Integer = 2
// CHECK: Integer = 0
// CHECK: Integer = 5
// CHECK-FIRST: [LambdaSpec] Replacing slot 3 with i32 2
// CHECK-FIRST: [LambdaSpec] Replacing slot 2 with i32 1
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 0
// CHECK: Integer = 0
// CHECK: Integer = 1
// CHECK: Integer = 2
// CHECK: Integer = 4
// CHECK: Integer = 0
// CHECK: Integer = 1
// CHECK: Integer = 2
// CHECK: Integer = 4
// CHECK-FIRST: [proteus][JitEngineHost] MemoryCache rank 0 hits 1 accesses 5
// CHECK: [proteus][JitEngineHost] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 2 NumHits 1
// CHECK-COUNT-3: [proteus][JitEngineHost] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: [proteus][JitEngineHost] StorageCache rank 0 hits 0 accesses 4
// CHECK-SECOND: [proteus][JitEngineHost] StorageCache rank 0 hits 4 accesses 4
// clang-format on
