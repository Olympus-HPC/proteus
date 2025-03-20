// RUN: rm -rf .proteus
// RUN: ./frontend | FileCheck %s --check-prefixes=CHECK
// RUN: rm -rf .proteus

#include <iostream>

#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

int main(int argc, char **argv) {
  proteus::init();

  auto J = proteus::JitModule();
  auto &F = J.addFunction<double, double>("MyFunc");
  auto &A = F.declVar<double>("a");
  auto &B = F.declVar<double>("b");
  auto &Arg = F.arg(0);
  A = 1;
  B = 2;
  A = A + B + Arg;
  // A = 4
  F.beginIf(A > 1.0);
  {
    F.beginIf(A > 2.0);
    {
      // A = 5
      A = A + 1.0;
      F.endIf();
    }
    // A = 5 or 6
    A = A + 1.0;
    F.endIf();
  }
  F.ret(A);
  J.print();
  double (*Func)(double) = (double (*)(double))J.compile();
  double Ret = Func(1.0);
  std::cout << "Ret " << Ret << "\n";

  proteus::finalize();
  return 0;
}

// CHECK: Ret 3
