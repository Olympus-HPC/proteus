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
      J.addFunction<void(float*, float*, float*, float*, float*, float*)>(
          "intrinsics");

  auto &PowOut = F.getArg<0>();
  auto &SqrtOut = F.getArg<1>();
  auto &ExpOut = F.getArg<2>();
  auto &SinOut = F.getArg<3>();
  auto &CosOut = F.getArg<4>();
  auto &FabsOut = F.getArg<5>();

  F.beginFunction();
  {
    // powf(float, int)
    auto B = F.declVar<float>();
    auto E = F.declVar<int>();
    B = 2.0f;
    E = 3;
    PowOut[0] = powf(B, E);

    // sqrtf(int)
    auto X = F.declVar<int>();
    X = 9;
    SqrtOut[0] = sqrtf(X);

    // expf(int)
    auto Y = F.declVar<int>();
    Y = 1;
    ExpOut[0] = expf(Y);

    // sinf(0.0f) and cosf(0.0f)
    auto Z = F.declVar<float>();
    Z = 0.0f;
    SinOut[0] = sinf(Z);
    CosOut[0] = cosf(Z);

    // fabs(int)
    auto W = F.declVar<int>();
    W = -5;
    FabsOut[0] = fabs(W);

    F.ret();
  }
  F.endFunction();

  J.compile();

  float RPow, RSqrt, RExp, RSin, RCos, RFabs;
  F(&RPow, &RSqrt, &RExp, &RSin, &RCos, &RFabs);

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
// CHECK: pow = 8
// CHECK-NEXT: sqrt = 3
// CHECK-NEXT: exp = 2.71828
// CHECK-NEXT: sin = 0
// CHECK-NEXT: cos = 1
// CHECK-NEXT: fabs = 5
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
// clang-format on
