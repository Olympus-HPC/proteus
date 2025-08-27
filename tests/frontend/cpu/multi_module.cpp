// clang-format off
// RUN: rm -rf .proteus
// RUN: ./multi_module | %FILECHECK %s --check-prefixes=CHECK
// RUN: rm -rf .proteus
// clang-format on

#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

using namespace proteus;

auto createJitModule1() {
  auto J = std::make_unique<JitModule>("host");

  auto &F1 = J->addFunction<void, double *>("f1");
  {
    F1.beginFunction();
    {
      auto [V] = F1.getArgs();
      V = 42;

      F1.ret();
    }
    F1.endFunction();
  }

  auto &F2 = J->addFunction<void, double *>("f2");
  {

    F2.beginFunction();
    {
      auto [V] = F2.getArgs();
      V = 23;

      F2.ret();
    }
    F2.endFunction();
  }

  return std::make_tuple(std::move(J), std::ref(F1), std::ref(F2));
}

auto createJitModule2() {
  auto J = std::make_unique<JitModule>("host");

  auto &F1 = J->addFunction<void, double *>("f1");
  {
    F1.beginFunction();
    {
      auto [V] = F1.getArgs();
      V = 142;

      F1.ret();
    }
    F1.endFunction();
  }

  auto &F2 = J->addFunction<void, double *>("f2");
  {
    F2.beginFunction();
    {
      auto [V] = F2.getArgs();
      V = 123;

      F2.ret();
    }
    F2.endFunction();
  }

  return std::make_tuple(std::move(J), std::ref(F1), std::ref(F2));
}

int main() {
  auto [J1, F11, F12] = createJitModule1();
  auto [J2, F21, F22] = createJitModule2();

  double V = 0;

  F11(&V);
  std::cout << "V " << V << "\n";

  F12(&V);
  std::cout << "V " << V << "\n";

  F21(&V);
  std::cout << "V " << V << "\n";

  F22(&V);
  std::cout << "V " << V << "\n";

  return 0;
}

// clang-format off
// CHECK: V 42
// CHECK-NEXT: V 23
// CHECK-NEXT: V 142
// CHECK-NEXT: V 123
// CHECK-NEXT: JitCache hits 0 total 0
