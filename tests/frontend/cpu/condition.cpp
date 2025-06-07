// RUN: rm -rf .proteus
// RUN: ./conditionals | FileCheck %s --check-prefixes=CHECK
// RUN: rm -rf .proteus

#include <iostream>

#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

int main(int argc, char **argv) {
  proteus::init();

  auto J = proteus::JitModule();
  auto &F = J.addFunction<double>("condition");
  auto &A = F.declVar<double>("a");
  auto &B = F.declVar<double>("b");

  auto &Cond = F.declVar<double>("cond");

  F.beginFunction();
  {
    A = 1;
    B = 2;
    F.beginIf(A < B);
    {
      A = 3;
    }
    F.endIf();
    F.ret(A);
  }
  F.endFunction();
  J.print();

  J.compile();
  double Ret = J.run<double>(F);
  std::cout << "Ret " << Ret << "\n";

  proteus::finalize();
  return 0;
}

// clang-format off

// CHECK: ; ModuleID = 'JitModule'
// CHECK-NEXT: source_filename = "JitModule"
// CHECK-NEXT: target triple = "{{.*}}"
// CHECK-EMPTY: 
// CHECK-LABEL: define double @Conditionals() {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %res. = alloca i1, align 1
// CHECK-NEXT:   %cond = alloca double, align 8
// CHECK-NEXT:   %b = alloca double, align 8
// CHECK-NEXT:   %a = alloca double, align 8
// CHECK-NEXT:   br label %body
// CHECK-EMPTY: 
// CHECK-NEXT: body:                                             ; preds = %entry
// CHECK-NEXT:   store double 1.000000e+00, ptr %a, align 8
// CHECK-NEXT:   store double 2.000000e+00, ptr %b, align 8
// CHECK-NEXT:   %0 = load double, ptr %a, align 8
// CHECK-NEXT:   %1 = load double, ptr %b, align 8
// CHECK-NEXT:   %2 = fcmp olt double %0, %1
// CHECK-NEXT:   store i1 %2, ptr %res., align 1
// CHECK-NEXT:   %3 = load i1, ptr %res., align 1
// CHECK-NEXT:   br i1 %3, label %if.then, label %cont
// CHECK-EMPTY: 
// CHECK-NEXT: if.then:                                          ; preds = %body
// CHECK-NEXT:   store double 3.000000e+00, ptr %a, align 8
// CHECK-NEXT:   br label %cont
// CHECK-EMPTY: 
// CHECK-NEXT: cont:                                             ; preds = %if.then, %body
// CHECK-NEXT:   %4 = load double, ptr %a, align 8
// CHECK-NEXT:   ret double %4
// CHECK-NEXT: }

// CHECK: Ret 3
