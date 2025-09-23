// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/for_early_ret | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/for_early_ret | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <iostream>

#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

int main() {
  proteus::init();

  auto J = proteus::JitModule();

  // Void early-terminate inside loop after i == 5.
  auto &FV = J.addFunction<void(double *)>("for_early_void");
  {
    auto &Arr = FV.getArg(0);
    auto &I = FV.declVar<int>("i");
    auto &Inc = FV.declVar<int>("inc");
    auto &UB = FV.declVar<int>("ub");

    FV.beginFunction();
    {
      I = 0;
      UB = 10;
      Inc = 1;
      FV.beginFor(I, I, UB, Inc);
      {
        Arr[I] = Arr[I] + 1.0;
        FV.beginIf(I == 5);
        { FV.ret(); }
        FV.endIf();
      }
      FV.endFor();
      FV.ret();
    }
    FV.endFunction();
  }

  // Non-void: return first index i == 5.
  auto &FN = J.addFunction<int()>("for_early_nonvoid");
  {
    auto &I = FN.declVar<int>("i");
    auto &Inc = FN.declVar<int>("inc");
    auto &UB = FN.declVar<int>("ub");

    FN.beginFunction();
    {
      I = 0;
      UB = 10;
      Inc = 1;
      FN.beginFor(I, I, UB, Inc);
      {
        FN.beginIf(I == 5);
        { FN.ret(I); }
        FN.endIf();
      }
      FN.endFor();
      // Should not be reached in this test, but keep a well-formed tail.
      auto &Ret = FN.declVar<int>("ret");
      Ret = -1;
      FN.ret(Ret);
    }
    FN.endFunction();
  }

  J.compile();

  double X[10];
  for (int I = 0; I < 10; I++) {
    X[I] = 1.0;
  }

  FV(X);
  for (int I = 0; I < 10; I++)
    std::cout << "X[" << I << "] = " << X[I] << "\n";

  int Idx = FN();
  std::cout << "IDX " << Idx << "\n";

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
// CHECK-NEXT: X[6] = 1
// CHECK-NEXT: X[7] = 1
// CHECK-NEXT: X[8] = 1
// CHECK-NEXT: X[9] = 1
// CHECK-NEXT: IDX 5
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
