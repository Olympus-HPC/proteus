// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/cast | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/cast | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <iostream>

#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

int main() {
  proteus::init();

  auto J = proteus::JitModule();
  auto &F = J.addFunction<void(double *, int *, float *, double *, int *,
                               float *, int *)>("cast");
  auto &DOut = F.getArg(0);
  auto &IOut = F.getArg(1);
  auto &FOut = F.getArg(2);
  auto &DOut2 = F.getArg(3);
  auto &IOut2 = F.getArg(4);
  auto &FOut2 = F.getArg(5);
  auto &IOut3 = F.getArg(6);

  F.beginFunction();
  {
    auto &Di = F.declVar<double>();
    auto &Ii = F.declVar<int>();
    auto &Fi = F.declVar<float>();

    Di = 3.9;
    Ii = -7;
    Fi = 2.5f;

    auto &IfromD = F.cast<int>(Di);
    auto &FfromI = F.cast<float>(Ii);
    auto &DfromF = F.cast<double>(Fi);

    DOut[0] = DfromF;
    IOut[0] = IfromD;
    FOut[0] = FfromI;

    auto &D2 = F.cast<double>(IfromD);
    auto &I2 = F.cast<int>(FfromI);
    auto &F2 = F.cast<float>(DfromF);

    DOut2[0] = D2;
    IOut2[0] = I2;
    FOut2[0] = F2;

    auto &I3 = F.cast<int>(Di);
    IOut3[0] = I3;

    F.ret();
  }
  F.endFunction();

  J.compile();

  double RD0, RD1;
  int RI0, RI1, RI2;
  float RF0, RF1;
  F(&RD0, &RI0, &RF0, &RD1, &RI1, &RF1, &RI2);

  std::cout << "RD0 = " << RD0 << "\n";
  std::cout << "RI0 = " << RI0 << "\n";
  std::cout << "RF0 = " << RF0 << "\n";
  std::cout << "RD1 = " << RD1 << "\n";
  std::cout << "RI1 = " << RI1 << "\n";
  std::cout << "RF1 = " << RF1 << "\n";
  std::cout << "RI2 = " << RI2 << "\n";

  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK: RD0 = 2.5
// CHECK-NEXT: RI0 = 3
// CHECK-NEXT: RF0 = -7
// CHECK-NEXT: RD1 = 3
// CHECK-NEXT: RI1 = -7
// CHECK-NEXT: RF1 = 2.5
// CHECK-NEXT: RI2 = 3
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
