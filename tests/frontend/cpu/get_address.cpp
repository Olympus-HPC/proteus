// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/get_address | %FILECHECK %s --check-prefixes=CHECK
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <iostream>

#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

using namespace proteus;

int main() {
  auto J = proteus::JitModule("host");
  {
    auto &Setter = J.addFunction<void(int *)>("set_via_ptr");
    Setter.beginFunction();
    {
      auto &P = Setter.getArg<0>();
      P[0] = 77;
      Setter.ret();
    }
    Setter.endFunction();
  }

  auto &F = J.addFunction<void(int *, int *)>("test_get_address");

  F.beginFunction();
  {
    auto &Out = F.getArg<0>();
    auto &In = F.getArg<1>();

    auto Scalar = F.declVar<int>("scalar");
    Scalar = 7;

    auto ScalarAddr = Scalar.getAddress();
    Out[0] = ScalarAddr[0];

    F.call<void(int *)>("set_via_ptr", ScalarAddr);
    Out[3] = Scalar;

    auto PtrAddr = In.getAddress();
    Out[1] = PtrAddr[0][0];

    PtrAddr[0][0] = 99;
    Out[2] = In[0];

    F.ret();
  }
  F.endFunction();

  J.compile();

  int Results[4] = {0, 0, 0, 0};
  int Input[1] = {5};
  F(Results, Input);

  std::cout << "getAddress scalar/pointer test:\n";
  std::cout << "  scalar via *: " << Results[0] << "\n";
  std::cout << "  pointer via **: " << Results[1] << "\n";
  std::cout << "  after store via **: " << Results[2] << "\n";
  std::cout << "  after internal set_via_ptr: " << Results[3] << "\n";

  return 0;
}

// clang-format off
// CHECK: getAddress scalar/pointer test:
// CHECK-NEXT:   scalar via *: 7
// CHECK-NEXT:   pointer via **: 5
// CHECK-NEXT:   after store via **: 99
// CHECK-NEXT:   after internal set_via_ptr: 77
