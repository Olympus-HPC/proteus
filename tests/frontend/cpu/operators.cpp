// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/operators | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/operators | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <iostream>

#include <proteus/JitFrontend.h>
#include <proteus/JitInterface.h>

using namespace proteus;

int main() {
  proteus::init();

  auto J = proteus::JitModule();
  auto &F = J.addFunction<void(
      const double *, const double *, double *, double *, double *, double *,
      double *, double *, double *, double *, double *, double *, double *,
      double *, double *, double *, double *, double *, double *, double *)>(
      "operators");

  auto [Arg0, Arg1, Arg2, Arg3, Arg4, Arg5, Arg6, Arg7, Arg8, Arg9, Arg10,
        Arg11, Arg12, Arg13, Arg14, Arg15, Arg16, Arg17, Arg18, Arg19] =
      F.getArgs();
  F.beginFunction();
  {
    Arg2[0] = Arg0[0] + Arg1[0];
    Arg3[0] = Arg0[0] - Arg1[0];
    Arg4[0] = Arg0[0] * Arg1[0];
    Arg5[0] = Arg0[0] / Arg1[0];
    *Arg6 = *Arg7 = *Arg8 = *Arg9 = 5;

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

    auto Cmp = F.declVar<double>("cmp");
    Cmp = 5.0;
    F.beginIf(Cmp <= 5.0);
    { Arg13[0] = 1.0; }
    F.endIf();

    auto ConstVal = F.defVar<const int>(Cmp, "const_val");
    Arg12[0] = Arg12[0] + ConstVal;

    auto ConstValScalar = F.defVar<const int>(1, "const_val_scalar");
    Arg12[0] = Arg12[0] + ConstValScalar;

    auto NotCond = !(Cmp <= 5.0);
    Arg15[0] = NotCond;

    Arg14[0] = -Arg0[0];

    // Check that Var<T*> operator* returns Var<T&>.
    Var<double &> Ref16 = *Arg16;
    Ref16 = 42.0;

    // Check that Var<T[]> operator[] returns Var<T&>.
    Var<double &> Ref17 = Arg17[0];
    Ref17 = 43.0;

    Var<double &> Ref18 = *Arg18;
    Ref18 = Ref16 + Ref17;

    // Test that getAddress() on Var<T&> returns the referenced pointer.
    auto Ref19 = *Arg19;
    auto Addr19 = Ref19.getAddress();
    auto RefAddr19 = *Addr19;
    RefAddr19 = 44.0;

    // Test that convert<const U> preserves the const qualifier.
    auto MutableVar = F.declVar<double>("mutable");
    MutableVar = 100.0;
    auto ConvertedConst = F.convert<const double>(MutableVar);
    static_assert(std::is_same_v<decltype(ConvertedConst), Var<const double>>);

    F.ret();
  }
  F.endFunction();

  J.compile();

  double R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15,
      R16, R17, R18, R19;
  R0 = 2.0;
  R1 = 3.0;
  R16 = R17 = R18 = R19 = 0.0;
  F(&R0, &R1, &R2, &R3, &R4, &R5, &R6, &R7, &R8, &R9, &R10, &R11, &R12, &R13,
    &R14, &R15, &R16, &R17, &R18, &R19);

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
  std::cout << "R14 = " << R14 << "\n";
  std::cout << "R15 = " << R15 << "\n";
  std::cout << "R16 = " << R16 << "\n";
  std::cout << "R17 = " << R17 << "\n";
  std::cout << "R18 = " << R18 << "\n";
  std::cout << "R19 = " << R19 << "\n";

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
// CHECK-NEXT: R12 = 13
// CHECK-NEXT: R13 = 1
// CHECK-NEXT: R14 = -2
// CHECK-NEXT: R15 = 0
// CHECK-NEXT: R16 = 42
// CHECK-NEXT: R17 = 43
// CHECK-NEXT: R18 = 85
// CHECK-NEXT: R19 = 44
// CHECK-FIRST: [proteus][DispatcherHost] StorageCache rank 0 hits 0 accesses 1
// CHECK-SECOND: [proteus][DispatcherHost] StorageCache rank 0 hits 1 accesses 1
