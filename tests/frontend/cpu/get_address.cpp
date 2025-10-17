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
  auto &F = J.addFunction<void(int *, double *)>("test_get_address");

  F.beginFunction();
  {
    auto &IntResult = F.getArg<0>();
    auto &DoubleResult = F.getArg<1>();

    auto IntVar = F.declVar<int>("int_var");
    IntVar = 42;

    auto IntAddr = IntVar.getAddress();
    IntResult[0] = IntAddr[0];

    auto DoubleVar = F.declVar<double>("double_var");
    DoubleVar = 3.14159;

    auto DoubleAddr = DoubleVar.getAddress();
    DoubleResult[0] = DoubleAddr[0];

    IntAddr[0] = 100;
    IntResult[1] = IntVar;

    DoubleAddr[0] = 2.71828;
    DoubleResult[1] = DoubleVar;

    F.ret();
  }
  F.endFunction();

  J.compile();

  int IntResults[2];
  double DoubleResults[2];

  F(IntResults, DoubleResults);

  std::cout << "ScalarVar<int> getAddress test:\n";
  std::cout << "  Original value: " << IntResults[0] << "\n";
  std::cout << "  Modified value: " << IntResults[1] << "\n";

  std::cout << "ScalarVar<double> getAddress test:\n";
  std::cout << "  Original value: " << DoubleResults[0] << "\n";
  std::cout << "  Modified value: " << DoubleResults[1] << "\n";

  return 0;
}

// clang-format off
// CHECK: ScalarVar<int> getAddress test:
// CHECK-NEXT:   Original value: 42
// CHECK-NEXT:   Modified value: 100
// CHECK: ScalarVar<double> getAddress test:
// CHECK-NEXT:   Original value: 3.14159
// CHECK-NEXT:   Modified value: 2.71828
