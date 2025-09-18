// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/for | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/for | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <iostream>

#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

int main() {
  proteus::init();

  auto J = proteus::JitModule();
  auto &F = J.addFunction<void(double *)>("for");

  auto &I = F.declVar<int>("i");
  auto &Inc = F.declVar<int>("inc");
  auto &UB = F.declVar<int>("ub");
  auto &Arg = F.getArg(0);
  F.beginFunction();
  {
    I = 0;
    UB = 10;
    Inc = 1;
    F.beginFor(I, I, UB, Inc);
    { Arg[I] = Arg[I] + 1.0; }
    F.endFor();
    F.ret();
  }
  F.endFunction();

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
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
