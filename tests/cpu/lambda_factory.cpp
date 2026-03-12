// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT=1 PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_factory.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT=1 PROTEUS_TRACE_OUTPUT="specialization" %build/lambda_factory.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <climits>
#include <cstdio>

#include <proteus/JitInterface.h>

template <typename LambdaType> class Abstraction {
public:
  LambdaType Lambda;
  // force a copy in the constructor
  Abstraction(LambdaType Lam) : Lambda(Lam){};
  // Abstraction(LambdaType& Lam) = delete;
  // Abstraction(const LambdaType& Lam) = delete;
  // Abstraction(LambdaType&& Lam) = delete;
  auto operator()() { return Lambda(); }
};

void printInt(int I) { printf("Integer = %d\n", I); }

template <typename T>
__attribute__((annotate("jit"))) void kernel(T LB) {
  LB();
}

namespace wrapper {
  template <typename T>
  auto jit_variable(T arg) { return proteus::jit_variable(arg); }
}

auto declareLambda(int rc1, int rc2) {
  return [=, C = proteus::jit_variable(rc1), D = 5, C2 = proteus::jit_variable(rc2)] ()
                         __attribute__((annotate("jit"))) { printInt(C); printInt(C2); printInt(D); };
}

auto declareLambdaThree(int rc1, int rc2, int rc3) {
  return [=, C = proteus::jit_variable(rc1), D=4, C2 = proteus::jit_variable(rc2), rc3= proteus::jit_variable(rc3)] ()
                         __attribute__((annotate("jit"))) { printInt(C); printInt(C2);printInt(rc3); printInt(D);};
}

auto forwardLambda(int rc1, int rc2, int rc3) {
  int loc = 1;
  return declareLambdaThree(rc1, loc, rc3);
}

template <typename T> void run_direct(T &&LB) {
  proteus::register_lambda(LB);
  LB();
}

template <typename T> void run_abs(T &&LB) {
  // proteus::register_lambda(LB);
  using DecayedT= std::decay_t<T>;
  Abstraction<DecayedT> Abs(LB);
  run_direct(Abs);
}


int main() {
  int Zero = 0;
  int One = 1;
  int Two = 2;

  auto ZeroLambda = declareLambda(Zero, One);

  auto OneLambda = declareLambda(One, Two);

  auto TwoLambda = declareLambda(Two, Zero);

  auto BigLam = declareLambdaThree(Zero, One, Two);

  auto forLam = forwardLambda(Zero, One, Two);
  std::cout <<"addr in main " << &ZeroLambda << "\n";
  std::cout <<"addr in main " << &OneLambda << "\n";
  std::cout <<"addr in main " << &TwoLambda << "\n";
  std::cout <<"addr in main " << &BigLam << "\n";
  std::cout <<"addr in main " << &forLam << "\n";
  run_abs(ZeroLambda);
  run_abs(OneLambda);
  run_direct(TwoLambda);
  run_direct(BigLam);
  run_abs(forLam);

  return 0;
}

// clang-format off
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 0
// CHECK: Integer = 0
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 1
// CHECK: Integer = 1
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 2
// CHECK: Integer = 2
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: [proteus][JitEngineDevice] StorageCache rank 0 hits 0 accesses 3
// CHECK-SECOND: [proteus][JitEngineDevice] StorageCache rank 0 hits 3 accesses 3
