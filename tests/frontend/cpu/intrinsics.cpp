// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/intrinsics | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/intrinsics | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <iostream>

#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

using namespace proteus;

int main() {

  auto J = JitModule();
  auto &F =
      J.addFunction<void(float *, float *, float *, float *, float *, int *,
                         float *, int *, float *, float *)>("intrinsics");
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
  F.beginFunction();
  {
    auto &X = F.declVar<float>();
    X = 2.0f;
    auto &Y = F.declVar<float>();
    Y = 3.0f;

    auto &Z = F.declVar<float>();
    Z = 9.0f;

    auto &W = F.declVar<float>();
    W = 0.0f;

    auto &A = F.declVar<float>();
    A = 1.0f;

    auto &F0 = F.declVar<float>();
    F0 = -1.0f;
    auto &F1 = F.declVar<float>();
    F1 = 4.5f;

    auto &I0 = F.declVar<int>();
    I0 = 7;
    auto &I1 = F.declVar<int>();
    I1 = -9;

    Arg0[0] = powf(X, Y);  // 8
    Arg1[0] = sqrtf(Z);    // 3
    Arg2[0] = expf(W);     // 1
    Arg3[0] = logf(A);     // 0
    Arg4[0] = min(F0, F1); // -1
    Arg5[0] = min(I0, I1); // -9
    Arg6[0] = max(F0, F1); // 4.5
    Arg7[0] = max(I0, I1); // 7
    Arg8[0] = absf(F0);    // 1

    auto &T = F.declVar<float>();
    T = -3.7f;
    Arg9[0] = truncf(T); // -3

    F.ret();
  }
  F.endFunction();

  J.compile();

  float R0, R1, R2, R3, R4, R6, R8, R9;
  int R5, R7;
  F(&R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7, &R8, &R9);

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

  return 0;
}

// clang-format off
// CHECK: R0 = 8
// CHECK-NEXT: R1 = 3
// CHECK-NEXT: R2 = 1
// CHECK-NEXT: R3 = 0
// CHECK-NEXT: R4 = -1
// CHECK-NEXT: R5 = -9
// CHECK-NEXT: R6 = 4.5
// CHECK-NEXT: R7 = 7
// CHECK-NEXT: R8 = 1
// CHECK-NEXT: R9 = -3
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
// clang-format on
