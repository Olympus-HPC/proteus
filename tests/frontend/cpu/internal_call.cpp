// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/internal_call | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/internal_call | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

using namespace proteus;

auto createJitModule1() {
  auto J = std::make_unique<JitModule>("host");

  auto &F1 = J->addFunction<void(double *)>("f1");
  {
    F1.beginFunction();
    {
      auto [V] = F1.getArgsTT();

      auto X = F1.defVar<double>(21);
      auto C = F1.callTT<double(void)>("f2");
      auto Res = F1.callTT<double(double, double)>("f3", X, C);
      V[0] = Res;

      F1.ret();
    }
    F1.endFunction();
  }

  auto &F2 = J->addFunction<double(void)>("f2");
  {
    F2.beginFunction();
    {
      auto C = F2.defVar<double>(2.0);
      F2.ret(C);
    }
    F2.endFunction();
  }

  auto &F3 = J->addFunction<double(double, double)>("f3");
  {
    F3.beginFunction();
    {
      auto [X, C] = F3.getArgsTT();
      auto P = F3.declVar<double>();
      P = X * C;
      F3.ret(P);
    }
    F3.endFunction();
  }

  return std::make_tuple(std::move(J), std::ref(F1), std::ref(F2));
}

int main() {
  auto [J1, F11, F12] = createJitModule1();

  double V = 0;

  F11(&V);
  std::cout << "V " << V << "\n";

  return 0;
}

// clang-format off
// CHECK: V 42
// CHECK: JitCache hits 0 total 1
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
