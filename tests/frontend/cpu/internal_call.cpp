// clang-format off
// RUN: rm -rf .proteus
// RUN: ./internal_call | %FILECHECK %s --check-prefixes=CHECK
// RUN: rm -rf .proteus
// clang-format on

#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

using namespace proteus;

// TODO: Defined functions across different modules must have different symbol
// names to avoid duplicate definitions, due to the singleton design for
// JitEngineHost and DispatcherHost. Reconsider.
auto createJitModule1() {
  auto J = std::make_unique<JitModule>("host");

  auto &F1 = J->addFunction<void, double *>("f1");
  {
    F1.beginFunction();
    {
      auto [V] = F1.getArgs();

      Var &Ret = F1.call<double>("f2");
      V[0] = Ret;

      F1.ret();
    }
    F1.endFunction();
  }

  auto &F2 = J->addFunction<double>("f2");
  {
    F2.beginFunction();
    {
      Var &V = F2.declVar<double>();
      V = 42;

      F2.ret(V);
    }
    F2.endFunction();
  }

  return std::make_tuple(std::move(J), std::ref(F1), std::ref(F2));
}

int main() {
  auto [J1, F11, F12] = createJitModule1();
  J1->print();

  double V = 0;

  F11(&V);
  std::cout << "V " << V << "\n";

  return 0;
}

// clang-format off
// CHECK: V 42
// CHECK-NEXT: JitCache hits 0 total 0
