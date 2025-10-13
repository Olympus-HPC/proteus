// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/convert | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/convert | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <iostream>

#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

int main() {
  proteus::init();

  auto J = proteus::JitModule();
  auto &F =
      J.addFunction<void(double *, int *, float *, double *, int *, float *)>(
          "cast");

  auto &DOut = F.getArgTT<0>();
  auto &IOut = F.getArgTT<1>();
  auto &FOut = F.getArgTT<2>();
  auto &DOut2 = F.getArgTT<3>();
  auto &IOut2 = F.getArgTT<4>();
  auto &FOut2 = F.getArgTT<5>();

  F.beginFunction();
  {
    auto Di = F.declVarTT<double>();
    auto Ii = F.declVarTT<int>();
    auto Fi = F.declVarTT<float>();

    Di = 3.9;
    Ii = -7;
    Fi = 2.5f;

    auto IfromD = F.convertTT<int>(Di);
    auto FfromI = F.convertTT<float>(Ii);
    auto DfromF = F.convertTT<double>(Fi);

    DOut[0] = DfromF;
    IOut[0] = IfromD;
    FOut[0] = FfromI;

    auto D2 = F.convertTT<double>(IfromD);
    auto I2 = F.convertTT<int>(FfromI);
    auto F2 = F.convertTT<float>(DfromF);

    DOut2[0] = D2;
    IOut2[0] = I2;
    FOut2[0] = F2;

    F.retTT();
  }
  F.endFunction();

  J.compile();

  double DoubleFromFloat, DoubleFromInt;
  int IntFromDouble, IntFromFloat;
  float FloatFromInt, FloatFromDouble;
  F(&DoubleFromFloat, &IntFromDouble, &FloatFromInt, &DoubleFromInt,
    &IntFromFloat, &FloatFromDouble);

  std::cout << "DoubleFromFloat = " << DoubleFromFloat << "\n";
  std::cout << "IntFromDouble = " << IntFromDouble << "\n";
  std::cout << "FloatFromInt = " << FloatFromInt << "\n";
  std::cout << "DoubleFromInt = " << DoubleFromInt << "\n";
  std::cout << "IntFromFloat = " << IntFromFloat << "\n";
  std::cout << "FloatFromDouble = " << FloatFromDouble << "\n";

  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK: DoubleFromFloat = 2.5
// CHECK-NEXT: IntFromDouble = 3
// CHECK-NEXT: FloatFromInt = -7
// CHECK-NEXT: DoubleFromInt = 3
// CHECK-NEXT: IntFromFloat = -7
// CHECK-NEXT: FloatFromDouble = 2.5
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
