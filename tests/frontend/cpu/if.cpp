// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/if | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/if | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <iostream>

#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

int main() {
  proteus::init();

  auto J = proteus::JitModule();
  auto &LT = J.addFunction<double(double, double)>("if.lt");
  {
    auto &Arg0 = LT.getArgTT<0>();
    auto &Arg1 = LT.getArgTT<1>();

    LT.beginFunction();
    {
      auto Ret = LT.declVarTT<double>("ret");
      Ret = 0;
      LT.beginIfTT(Arg0 < Arg1);
      { Ret = 1; }
      LT.endIf();
      LT.retTT(Ret);
    }
    LT.endFunction();
  }

  auto &LE = J.addFunction<double(double, double)>("if.le");
  {
    auto &Arg0 = LE.getArgTT<0>();
    auto &Arg1 = LE.getArgTT<1>();

    LE.beginFunction();
    {
      auto Ret = LE.declVarTT<double>("ret");
      Ret = 0;
      LE.beginIfTT(Arg0 <= Arg1);
      { Ret = 1; }
      LE.endIf();
      LE.retTT(Ret);
    }
    LE.endFunction();
  }

  auto &GT = J.addFunction<double(double, double)>("if.gt");
  {
    auto &Arg0 = GT.getArgTT<0>();
    auto &Arg1 = GT.getArgTT<1>();

    GT.beginFunction();
    {
      auto Ret = GT.declVarTT<double>("ret");
      Ret = 0;
      GT.beginIfTT(Arg0 > Arg1);
      { Ret = 1; }
      GT.endIf();
      GT.retTT(Ret);
    }
    GT.endFunction();
  }

  auto &GE = J.addFunction<double(double, double)>("if.ge");
  {
    auto &Arg0 = GE.getArgTT<0>();
    auto &Arg1 = GE.getArgTT<1>();

    GE.beginFunction();
    {
      auto Ret = GE.declVarTT<double>("ret");
      Ret = 0;
      GE.beginIfTT(Arg0 >= Arg1);
      { Ret = 1; }
      GE.endIf();
      GE.retTT(Ret);
    }
    GE.endFunction();
  }

  auto &EQ = J.addFunction<double(double, double)>("if.eq");
  {
    auto &Arg0 = EQ.getArgTT<0>();
    auto &Arg1 = EQ.getArgTT<1>();

    EQ.beginFunction();
    {
      auto Ret = EQ.declVarTT<double>("ret");
      Ret = 0;
      EQ.beginIfTT(Arg0 == Arg1);
      { Ret = 1; }
      EQ.endIf();
      EQ.retTT(Ret);
    }
    EQ.endFunction();
  }

  auto &NE = J.addFunction<double(double, double)>("if.ne");
  {
    auto &Arg0 = NE.getArgTT<0>();
    auto &Arg1 = NE.getArgTT<1>();

    NE.beginFunction();
    {
      auto Ret = NE.declVarTT<double>("ret");
      Ret = 0;
      NE.beginIfTT(Arg0 != Arg1);
      { Ret = 1; }
      NE.endIf();
      NE.retTT(Ret);
    }
    NE.endFunction();
  }

  J.compile();

  // LT tests
  // Evaluates to true.
  double Ret = LT(1.0, 2.0);
  std::cout << "R LT " << Ret << "\n";
  // Evaluates to false.
  Ret = LT(2.0, 1.0);
  std::cout << "R LT " << Ret << "\n";

  // LE tests
  // Evaluates to true.
  Ret = LE(1.0, 2.0);
  std::cout << "R LE " << Ret << "\n";
  // Evaluates to true (equal).
  Ret = LE(2.0, 2.0);
  std::cout << "R LE " << Ret << "\n";
  // Evaluates to false.
  Ret = LE(3.0, 2.0);
  std::cout << "R LE " << Ret << "\n";

  // GT tests
  // Evaluates to true.
  Ret = GT(3.0, 2.0);
  std::cout << "R GT " << Ret << "\n";
  // Evaluates to false.
  Ret = GT(2.0, 3.0);
  std::cout << "R GT " << Ret << "\n";

  // GE tests
  // Evaluates to true.
  Ret = GE(3.0, 2.0);
  std::cout << "R GE " << Ret << "\n";
  // Evaluates to true (equal).
  Ret = GE(2.0, 2.0);
  std::cout << "R GE " << Ret << "\n";
  // Evaluates to false.
  Ret = GE(1.0, 2.0);
  std::cout << "R GE " << Ret << "\n";

  // EQ tests
  // Evaluates to true.
  Ret = EQ(2.0, 2.0);
  std::cout << "R EQ " << Ret << "\n";
  // Evaluates to false.
  Ret = EQ(2.0, 3.0);
  std::cout << "R EQ " << Ret << "\n";

  // NE tests
  // Evaluates to true.
  Ret = NE(2.0, 3.0);
  std::cout << "R NE " << Ret << "\n";
  // Evaluates to false.
  Ret = NE(2.0, 2.0);
  std::cout << "R NE " << Ret << "\n";

  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK: R LT 1
// CHECK-NEXT: R LT 0
// CHECK-NEXT: R LE 1
// CHECK-NEXT: R LE 1
// CHECK-NEXT: R LE 0
// CHECK-NEXT: R GT 1
// CHECK-NEXT: R GT 0
// CHECK-NEXT: R GE 1
// CHECK-NEXT: R GE 1
// CHECK-NEXT: R GE 0
// CHECK-NEXT: R EQ 1
// CHECK-NEXT: R EQ 0
// CHECK-NEXT: R NE 1
// CHECK-NEXT: R NE 0
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
