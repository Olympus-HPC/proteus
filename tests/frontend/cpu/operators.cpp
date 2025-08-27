// RUN: rm -rf .proteus
// RUN: ./operators | %FILECHECK %s --check-prefixes=CHECK
// RUN: rm -rf .proteus

#include <iostream>

#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

int main() {
  proteus::init();

  auto J = proteus::JitModule();
  auto &F =
      J.addFunction<void, double *, double *, double *, double *, double *,
                    double *, double *, double *, double *, double *, double *,
                    double *>(
          "operators");
  auto &Arg0 = F.getArg(0);
  auto &Arg1 = F.getArg(1);
  auto &Arg2 = F.getArg(2);
  auto &Arg3 = F.getArg(3);
  auto &Arg4 = F.getArg(4);
  auto &Arg5 = F.getArg(5);
  auto &Arg6 = F.getArg(6);
  auto &Arg7 = F.getArg(7);
  auto &Arg8 = F.getArg(8);
  auto &Arg9 = F.getArg(9);
  auto &Arg10 = F.getArg(10);
  auto &Arg11 = F.getArg(11);
  F.beginFunction();
  {
    Arg0[0] = 2;
    Arg1[0] = 3;

    Arg2[0] = Arg0 + Arg1;
    Arg3[0] = Arg0 - Arg1;
    Arg4[0] = Arg0 * Arg1;
    Arg5[0] = Arg0 / Arg1;

    Arg6[0] = Arg7[0] = Arg8[0] = Arg9[0] = Arg10[0] = Arg11[0] = 5;

    Arg6[0] += Arg0[0];
    Arg7[0] -= Arg0[0];
    Arg8[0] *= Arg0[0];
    Arg9[0] /= Arg0[0];

    Arg10[0] = Arg0 % Arg1;
    Arg11[0] %= Arg0[0];

    F.ret();
  }
  F.endFunction();

  J.compile();

  double R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11;
  F(&R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7, &R8, &R9, &R10, &R11);

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
