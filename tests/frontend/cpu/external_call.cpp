// RUN: rm -rf .proteus
// RUN: ./external_call | FileCheck %s --check-prefixes=CHECK
// RUN: rm -rf .proteus

#include <iostream>

#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

extern "C" {
void hello() { std::cout << "Hello!\n"; }
}

int main(int argc, char **argv) {
  proteus::init();

  auto J = proteus::JitModule();
  auto &F = J.addFunction<void>("ExternalCall");
  F.beginFunction();
  {
    F.call<void>("hello");
  }
  F.ret();
  F.endFunction();

  J.print();
  J.compile();
  J.run<void>(F);

  proteus::finalize();
  return 0;
}

// clang-format off

//CHECK-LABEL: ; ModuleID = 'JitModule'
//CHECK-NEXT: source_filename = "JitModule"
//CHECK-NEXT: target triple = {{.*}}
//CHECK-EMPTY: 
//CHECK-NEXT: define void @ExternalCall() {
//CHECK-NEXT: entry:
//CHECK-NEXT:   br label %body
//CHECK-EMPTY: 
//CHECK-NEXT: body:                                             ; preds = %entry
//CHECK-NEXT:   call void @hello()
//CHECK-NEXT:   ret void
//CHECK-NEXT: }
//CHECK-EMPTY:
//CHECK-NEXT: declare void @hello()
//CHECK-NEXT: Hello!