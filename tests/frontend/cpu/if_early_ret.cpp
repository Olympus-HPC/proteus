// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/if_early_ret | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/if_early_ret | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <iostream>

#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

int main() {
  proteus::init();

  auto J = proteus::JitModule();

  // Non-void with early-return in IF.
  auto &F = J.addFunction<double(double, double)>("if_early_nonvoid");
  {
    auto &A = F.getArg(0);
    auto &B = F.getArg(1);

    F.beginFunction();
    {
      auto &Ret = F.declVar<double>("ret");
      Ret = 0;
      F.beginIf(A < B);
      { Ret = 1; F.ret(Ret); }
      F.endIf();
      F.ret(Ret);
    }
    F.endFunction();
  }

  // Void with early-return in IF guarding a side effect.
  auto &G = J.addFunction<void(double *, double, double)>("if_early_void");
  {
    auto &Arr = G.getArg(0);
    auto &A = G.getArg(1);
    auto &B = G.getArg(2);

    auto &I = G.declVar<int>("i");

    G.beginFunction();
    {
      I = 0;
      Arr[I] = 0.0;
      G.beginIf(A < B);
      { G.ret(); }
      G.endIf();
      Arr[I] = Arr[I] + 1.0;
      G.ret();
    }
    G.endFunction();
  }

  J.compile();

  // Non-void checks
  double Ret = F(1.0, 2.0);
  std::cout << "R IF-EARLY " << Ret << "\n";
  Ret = F(2.0, 1.0);
  std::cout << "R IF-EARLY " << Ret << "\n";

  // Void checks
  double X = 0.0;
  G(&X, 1.0, 2.0);
  std::cout << "X " << X << "\n";
  X = 0.0;
  G(&X, 2.0, 1.0);
  std::cout << "X " << X << "\n";

  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK: R IF-EARLY 1
// CHECK-NEXT: R IF-EARLY 0
// CHECK-NEXT: X 0
// CHECK-NEXT: X 1
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
