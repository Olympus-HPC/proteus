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
  proteus::init();

  auto J = JitModule();
  auto &F =
      J.addFunction<void(float *, float *, float *, float *, float *, float *,
                         float *, int *, float *, float *, float *, int *)>(
          "intrinsics");
  auto &PowOut = F.getArg<0>();
  auto &SqrtOut = F.getArg<1>();
  auto &ExpOut = F.getArg<2>();
  auto &SinOut = F.getArg<3>();
  auto &CosOut = F.getArg<4>();
  auto &FabsOut = F.getArg<5>();
  auto &PowBase = F.getArg<6>();
  auto &PowExp = F.getArg<7>();
  auto &SqrtIn = F.getArg<8>();
  auto &ExpIn = F.getArg<9>();
  auto &TrigIn = F.getArg<10>();
  auto &FabsIn = F.getArg<11>();

  F.beginFunction();
  {
    PowOut[0] = powf(PowBase[0], PowExp[0]);

    auto SqrtVal = SqrtIn[0];
    SqrtOut[0] = sqrtf(SqrtVal);
    ExpOut[0] = expf(ExpIn[0]);

    auto TrigInput = TrigIn[0];
    SinOut[0] = sinf(TrigInput);
    CosOut[0] = cosf(TrigInput);

    FabsOut[0] = fabs(FabsIn[0]);

    F.ret();
  }
  F.endFunction();

  J.compile();

  float RPow, RSqrt, RExp, RSin, RCos, RFabs;
  float PowBaseBuf, SqrtInVal, ExpInVal, TrigInVal;
  int PowExponentBuf, FabsInVal;

  PowBaseBuf = 1.5f;
  PowExponentBuf = 5;
  SqrtInVal = 50.0f;
  ExpInVal = 0.7f;
  TrigInVal = 0.7f;
  FabsInVal = -12;

  F(&RPow, &RSqrt, &RExp, &RSin, &RCos, &RFabs, &PowBaseBuf, &PowExponentBuf,
    &SqrtInVal, &ExpInVal, &TrigInVal, &FabsInVal);

  std::cout << "pow = " << RPow << "\n";
  std::cout << "sqrt = " << RSqrt << "\n";
  std::cout << "exp = " << RExp << "\n";
  std::cout << "sin = " << RSin << "\n";
  std::cout << "cos = " << RCos << "\n";
  std::cout << "fabs = " << RFabs << "\n";

  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK: pow = 7.59375
// CHECK-NEXT: sqrt = 7.07107
// CHECK-NEXT: exp = 2.01375
// CHECK-NEXT: sin = 0.644218
// CHECK-NEXT: cos = 0.764842
// CHECK-NEXT: fabs = 12
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
// clang-format on
