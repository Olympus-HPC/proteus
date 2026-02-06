// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/while | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/while | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <iostream>

#include <proteus/JitFrontend.h>

int main() {
  proteus::init();

  auto J = proteus::JitModule();
  auto &F = J.addFunction<void(double *)>("while");

  auto I = F.declVar<int>("i");
  auto N = F.declVar<int>("n");
  auto &Arg = F.getArg<0>();

  F.beginFunction();
  {
    I = 0;
    N = 10;
    F.beginWhile([&]() { return I < N; });
    {
      Arg[I] = Arg[I] + I;
      I = I + 1;
    }
    F.endWhile();
    F.ret();
  }
  F.endFunction();

  J.compile();

  double X[10];
  for (int I = 0; I < 10; I++)
    X[I] = 1.0;
  F(X);
  for (int I = 0; I < 10; I++)
    std::cout << "X[" << I << "] = " << X[I] << "\n";

  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK: X[0] = 1
// CHECK-NEXT: X[1] = 2
// CHECK-NEXT: X[2] = 3
// CHECK-NEXT: X[3] = 4
// CHECK-NEXT: X[4] = 5
// CHECK-NEXT: X[5] = 6
// CHECK-NEXT: X[6] = 7
// CHECK-NEXT: X[7] = 8
// CHECK-NEXT: X[8] = 9
// CHECK-NEXT: X[9] = 10
// CHECK-FIRST: [proteus][DispatcherHost] StorageCache rank 0 hits 0 accesses 1
// CHECK-SECOND: [proteus][DispatcherHost] StorageCache rank 0 hits 1 accesses 1
