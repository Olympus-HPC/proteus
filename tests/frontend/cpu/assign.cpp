// RUN: rm -rf .proteus
// RUN: ./assignments| FileCheck %s --check-prefixes=CHECK
// RUN: rm -rf .proteus

#include <iostream>

#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

int main(int argc, char **argv) {
  proteus::init();

  auto J = proteus::JitModule();
  auto &F = J.addFunction<double, double *>("assign");
  auto &A = F.declVar<double>("a");
  auto &Arg = F.getArg(0);
  F.beginFunction();
  {
    A = 1;
    Arg[0] = A + 42;
    F.ret(A);
  }
  F.endFunction();

  auto &F2 = J.addFunction<double>("test");
  auto &R = F2.declVar<double>("ret");
  F2.beginFunction();
  {
    R = 23.0;
    F2.ret(R);
  }
  F2.endFunction();

  J.print();
  J.compile();
  double X[1] = {1.0};
  double Ret = 123;
  J.run<double>(F, &X[0]);
  std::cout << "Ret " << Ret << "\n";
  std::cout << "X[0] = " << X[0] << "\n";

  double R2 = J.run<double>(F2);
  std::cout << "F2 ret " << R2 << "\n";

  proteus::finalize();
  return 0;
}

// clang-format off

// CHECK: ; ModuleID = 'JitModule'
// CHECK: source_filename = "JitModule"
// CHECK: target triple = "arm64-apple-darwin22.6.0"
// CHECK-EMPTY: 
// CHECK-LABEL: define double @Assignments(ptr %0) {
// CHECK: entry:
// CHECK:   %res. = alloca ptr, align 8
// CHECK:   %a = alloca double, align 8
// CHECK:   %arg.0 = alloca ptr, align 8
// CHECK:   store ptr %0, ptr %arg.0, align 8
// CHECK:   br label %body
// CHECK-EMPTY: 
// CHECK: body:                                             ; preds = %entry
// CHECK:   store double 1.000000e+00, ptr %a, align 8
// CHECK:   %1 = load ptr, ptr %arg.0, align 8
// CHECK:   %2 = getelementptr inbounds double, ptr %1, i64 0
// CHECK:   store ptr %2, ptr %res., align 8
// CHECK:   %3 = load ptr, ptr %res., align 8
// CHECK:   store double 4.200000e+01, ptr %3, align 8
// CHECK:   %4 = load double, ptr %a, align 8
// CHECK:   ret double %4
// CHECK: }

// CHECK: Ret 1
// CHECK: X[0] = 42
