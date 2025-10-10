// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/operators | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/operators | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <iostream>

#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

int main() {
  proteus::init();

  auto J = proteus::JitModule();
  auto &F =
      J.addFunction<void(double *, double *, double *, double *, double *,
                         double *, double *, double *, double *, double *,
                         double *, double *, double *, double *)>("operators");

  auto [Arg0, Arg1, Arg2, Arg3, Arg4, Arg5, Arg6, Arg7, Arg8, Arg9, Arg10, Arg11, Arg12, Arg13] = F.getArgsTT();
  F.beginFunction();
  {
    Arg0[0] = 2;
    Arg1[0] = 3;

    Arg2[0] = Arg0[0] + Arg1[0];
    Arg3[0] = Arg0[0] - Arg1[0];
    Arg4[0] = Arg0[0] * Arg1[0];
    Arg5[0] = Arg0[0] / Arg1[0];
    *Arg6 = *Arg7 = *Arg8 = *Arg9 = 5;
    // *Arg2[0] = Arg0 + Arg1;

    Arg6[0] += Arg0[0];
    Arg7[0] -= Arg0[0];
    Arg8[0] *= Arg0[0];
    Arg9[0] /= Arg0[0];

    Arg10[0] = Arg0[0] % Arg1[0];
    Arg11[0] = 5;
    Arg11[0] %= Arg0[0];

    Arg12[0] = 10.0;
    Arg12[0] -= 3.0;

    Arg13[0] = 1.0;

    auto Cmp = F.declVarTT<double>("cmp");
    Cmp = 5.0;
    F.beginIfTT(Cmp <= 5.0);
    { Arg13[0] = 1.0; }
    F.endIf();


    F.ret();
  }
  F.endFunction();

  J.compile();

  double R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13 = 0.0;
  R0 = 5.0;
  R1 = 3.0;
  F(&R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7, &R8, &R9, &R10, &R11, &R12, &R13);

  std::cout << "R0 = " << R0 << "\n";
  std::cout << "R1 = " << R1 << "\n";
  std::cout << "R2 = " << R2 << "\n";
  std::cout << "R3 = " << R3 << "\n";
  std::cout << "R4 = " << R4 << "\n";
  std::cout << "R5 = " << R5 << "\n";
  std::cout << "R6 = " << R6 << "\n";
  std::cout << "R7 = " << R7 << "\n";
  std::cout << "R8 = " << R8 << "\n";
  std::cout << "R9 = " << R9 << "\n";
  std::cout << "R10 = " << R10 << "\n";
  std::cout << "R11 = " << R11 << "\n";
  std::cout << "R12 = " << R12 << "\n";
  std::cout << "R13 = " << R13 << "\n";

  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK: R0 = 2
// CHECK-NEXT: R1 = 3
// CHECK-NEXT: R2 = 5
// CHECK-NEXT: R3 = -1
// CHECK-NEXT: R4 = 6
// CHECK-NEXT: R5 = 0.666667
// CHECK-NEXT: R6 = 7
// CHECK-NEXT: R7 = 3
// CHECK-NEXT: R8 = 10
// CHECK-NEXT: R9 = 2.5
// CHECK-NEXT: R10 = 2
// CHECK-NEXT: R11 = 1
// CHECK-NEXT: R12 = 7
// CHECK-NEXT: R13 = 1
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
