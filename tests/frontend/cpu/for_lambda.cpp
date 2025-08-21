// RUN: rm -rf .proteus
// RUN: ./for_lambda | %FILECHECK %s --check-prefixes=CHECK
// RUN: rm -rf .proteus

#include <iostream>

#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

int main() {
  proteus::init();

  auto J = proteus::JitModule();
  auto &F = J.addFunction<void, double *>("for_lambda");

  auto &I = F.declVar<int>("i");
  auto &Inc = F.declVar<int>("inc");
  auto &UB = F.declVar<int>("ub");
  auto &Arg = F.getArg(0);
  F.function([&]() {
    I = 0;
    UB = 10;
    Inc = 1;
    F.For(I, I, UB, Inc, [&]() {
      Arg[I] = Arg[I] + 1.0;
    });
    F.ret();
  });

  J.compile();

  double X[10];
  for (int I = 0; I < 10; I++) {
    X[I] = 1.0;
  }

  F(X);
  for (int I = 0; I < 10; I++)
    std::cout << "X[" << I << "] = " << X[I] << "\n";

  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK: X[0] = 2
// CHECK-NEXT: X[1] = 2
// CHECK-NEXT: X[2] = 2
// CHECK-NEXT: X[3] = 2
// CHECK-NEXT: X[4] = 2
// CHECK-NEXT: X[5] = 2
// CHECK-NEXT: X[6] = 2
// CHECK-NEXT: X[7] = 2
// CHECK-NEXT: X[8] = 2
// CHECK-NEXT: X[9] = 2

