// RUN: rm -rf .proteus
// RUN: ./if | %FILECHECK %s --check-prefixes=CHECK
// RUN: rm -rf .proteus

#include <iostream>

#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

int main() {
  proteus::init();

  auto J = proteus::JitModule();
  auto &LT = J.addFunction<double, double, double>("if.lt");
  {
    auto &Arg0 = LT.getArg(0);
    auto &Arg1 = LT.getArg(1);

    LT.beginFunction();
    {
      auto &Ret = LT.declVar<double>("ret");
      Ret = 0;
      LT.beginIf(Arg0 < Arg1);
      { Ret = 1; }
      LT.endIf();
      LT.ret(Ret);
    }
    LT.endFunction();
  }

  auto &LE = J.addFunction<double, double, double>("if.le");
  {
    auto &Arg0 = LE.getArg(0);
    auto &Arg1 = LE.getArg(1);

    LE.beginFunction();
    {
      auto &Ret = LE.declVar<double>("ret");
      Ret = 0;
      LE.beginIf(Arg0 <= Arg1);
      { Ret = 1; }
      LE.endIf();
      LE.ret(Ret);
    }
    LE.endFunction();
  }

  auto &GT = J.addFunction<double, double, double>("if.gt");
  {
    auto &Arg0 = GT.getArg(0);
    auto &Arg1 = GT.getArg(1);

    GT.beginFunction();
    {
      auto &Ret = GT.declVar<double>("ret");
      Ret = 0;
      GT.beginIf(Arg0 > Arg1);
      { Ret = 1; }
      GT.endIf();
      GT.ret(Ret);
    }
    GT.endFunction();
  }

  auto &GE = J.addFunction<double, double, double>("if.ge");
  {
    auto &Arg0 = GE.getArg(0);
    auto &Arg1 = GE.getArg(1);

    GE.beginFunction();
    {
      auto &Ret = GE.declVar<double>("ret");
      Ret = 0;
      GE.beginIf(Arg0 >= Arg1);
      { Ret = 1; }
      GE.endIf();
      GE.ret(Ret);
    }
    GE.endFunction();
  }

  auto &EQ = J.addFunction<double, double, double>("if.eq");
  {
    auto &Arg0 = EQ.getArg(0);
    auto &Arg1 = EQ.getArg(1);

    EQ.beginFunction();
    {
      auto &Ret = EQ.declVar<double>("ret");
      Ret = 0;
      EQ.beginIf(Arg0 == Arg1);
      { Ret = 1; }
      EQ.endIf();
      EQ.ret(Ret);
    }
    EQ.endFunction();
  }

  auto &NE = J.addFunction<double, double, double>("if.ne");
  {
    auto &Arg0 = NE.getArg(0);
    auto &Arg1 = NE.getArg(1);

    NE.beginFunction();
    {
      auto &Ret = NE.declVar<double>("ret");
      Ret = 0;
      NE.beginIf(Arg0 != Arg1);
      { Ret = 1; }
      NE.endIf();
      NE.ret(Ret);
    }
    NE.endFunction();
  }

  J.compile();

  // LT tests
  // Evaluates to true.
  double Ret = J.run<double, double, double>(LT, 1.0, 2.0);
  std::cout << "R LT " << Ret << "\n";
  // Evaluates to false.
  Ret = J.run<double, double, double>(LT, 2.0, 1.0);
  std::cout << "R LT " << Ret << "\n";

  // LE tests
  // Evaluates to true.
  Ret = J.run<double, double, double>(LE, 1.0, 2.0);
  std::cout << "R LE " << Ret << "\n";
  // Evaluates to true (equal).
  Ret = J.run<double, double, double>(LE, 2.0, 2.0);
  std::cout << "R LE " << Ret << "\n";
  // Evaluates to false.
  Ret = J.run<double, double, double>(LE, 3.0, 2.0);
  std::cout << "R LE " << Ret << "\n";

  // GT tests
  // Evaluates to true.
  Ret = J.run<double, double, double>(GT, 3.0, 2.0);
  std::cout << "R GT " << Ret << "\n";
  // Evaluates to false.
  Ret = J.run<double, double, double>(GT, 2.0, 3.0);
  std::cout << "R GT " << Ret << "\n";

  // GE tests
  // Evaluates to true.
  Ret = J.run<double, double, double>(GE, 3.0, 2.0);
  std::cout << "R GE " << Ret << "\n";
  // Evaluates to true (equal).
  Ret = J.run<double, double, double>(GE, 2.0, 2.0);
  std::cout << "R GE " << Ret << "\n";
  // Evaluates to false.
  Ret = J.run<double, double, double>(GE, 1.0, 2.0);
  std::cout << "R GE " << Ret << "\n";

  // EQ tests
  // Evaluates to true.
  Ret = J.run<double, double, double>(EQ, 2.0, 2.0);
  std::cout << "R EQ " << Ret << "\n";
  // Evaluates to false.
  Ret = J.run<double, double, double>(EQ, 2.0, 3.0);
  std::cout << "R EQ " << Ret << "\n";

  // NE tests
  // Evaluates to true.
  Ret = J.run<double, double, double>(NE, 2.0, 3.0);
  std::cout << "R NE " << Ret << "\n";
  // Evaluates to false.
  Ret = J.run<double, double, double>(NE, 2.0, 2.0);
  std::cout << "R NE " << Ret << "\n";

  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK: R LT 1
// CHECK: R LT 0
// CHECK: R LE 1
// CHECK: R LE 1
// CHECK: R LE 0
// CHECK: R GT 1
// CHECK: R GT 0
// CHECK: R GE 1
// CHECK: R GE 1
// CHECK: R GE 0
// CHECK: R EQ 1
// CHECK: R EQ 0
// CHECK: R NE 1
// CHECK: R NE 0
// CHECK: JitCache hits 0 total 0
// CHECK: JitStorageCache hits 0 total 0
// CHECK: JitCache hits 0 total 0
