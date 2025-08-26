// RUN: rm -rf .proteus
// RUN: ./min | %FILECHECK %s --check-prefixes=CHECK
// RUN: rm -rf .proteus

#include <iostream>

#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

using namespace proteus;

int main() {
  proteus::init();

  auto J = JitModule();
  auto &F = J.addFunction<void, float *, float *, float *, int *, int *, int *>(
      "min_test");

  auto &Ff0 = F.getArg(0);
  auto &Ff1 = F.getArg(1);
  auto &FfRes = F.getArg(2);

  auto &Fi0 = F.getArg(3);
  auto &Fi1 = F.getArg(4);
  auto &FiRes = F.getArg(5);

  F.beginFunction();
  {
    Ff0[0] = 2.5f;
    Ff1[0] = -3.0f;
    FfRes[0] = min(Ff0, Ff1);

    Fi0[0] = 7;
    Fi1[0] = 9;
    FiRes[0] = min(Fi0, Fi1);

    F.ret();
  }
  F.endFunction();

  J.compile();

  float rf0, rf1, rfRes;
  int ri0, ri1, riRes;
  F(&rf0, &rf1, &rfRes, &ri0, &ri1, &riRes);

  std::cout << "rf0 = " << rf0 << "\n";
  std::cout << "rf1 = " << rf1 << "\n";
  std::cout << "rfRes = " << rfRes << "\n";
  std::cout << "ri0 = " << ri0 << "\n";
  std::cout << "ri1 = " << ri1 << "\n";
  std::cout << "riRes = " << riRes << "\n";

  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK: rf0 = 2.5
// CHECK-NEXT: rf1 = -3
// CHECK-NEXT: rfRes = -3
// CHECK-NEXT: ri0 = 7
// CHECK-NEXT: ri1 = 9
// CHECK-NEXT: riRes = 7
// clang-format on


