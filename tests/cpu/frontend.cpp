// RUN: rm -rf .proteus
// RUN: ./frontend | FileCheck %s --check-prefixes=CHECK
// RUN: rm -rf .proteus

#include <iostream>

#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

int main(int argc, char **argv) {
  proteus::init();

  auto J = proteus::JitModule();
  auto F = J.addFunction<proteus::LLVMTypeMap<double>>("MyFunc");
  auto &A = J.addVariable<proteus::LLVMTypeMap<double>>("a", F);
  auto &B = J.addVariable<proteus::LLVMTypeMap<double>>("b", F);
  A = 1;
  B = 2;
  A = A + B;
  F.addRet(A);
  J.print();
  double (*Func)(double) = (double (*)(double))J.compile();
  double Ret = Func(0.0);
  std::cout << "Ret " << Ret << "\n";

  proteus::finalize();
  return 0;
}

// CHECK: Ret 3
