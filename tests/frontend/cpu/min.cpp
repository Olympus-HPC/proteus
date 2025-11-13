// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/min | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/min | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <iostream>

#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

using namespace proteus;

int main() {
  proteus::init();

  auto J = JitModule();
  auto &F = J.addFunction<void(float *, float *, float *, int *, int *, int *)>(
      "min_test");

  auto &Ff0 = F.getArg<0>();
  auto &Ff1 = F.getArg<1>();
  auto &FfRes = F.getArg<2>();

  auto &Fi0 = F.getArg<3>();
  auto &Fi1 = F.getArg<4>();
  auto &FiRes = F.getArg<5>();

  F.beginFunction();
  {
    Ff0[0] = 2.5f;
    Ff1[0] = -3.0f;
    FfRes[0] = min(*Ff0, *Ff1);

    Fi0[0] = 7;
    Fi1[0] = 9;
    FiRes[0] = min(*Fi0, *Fi1);

    F.ret();
  }
  F.endFunction();

  J.compile();

  float Rf0, Rf1, RfRes;
  int Ri0, Ri1, RiRes;
  F(&Rf0, &Rf1, &RfRes, &Ri0, &Ri1, &RiRes);

  std::cout << "rf0 = " << Rf0 << "\n";
  std::cout << "rf1 = " << Rf1 << "\n";
  std::cout << "rfRes = " << RfRes << "\n";
  std::cout << "ri0 = " << Ri0 << "\n";
  std::cout << "ri1 = " << Ri1 << "\n";
  std::cout << "riRes = " << RiRes << "\n";

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
// CHECK-FIRST: [proteus][DispatcherHost] StorageCache procuid 0 hits 0 accesses 1
// CHECK-SECOND: [proteus][DispatcherHost] StorageCache procuid 0 hits 1 accesses 1
// clang-format on
