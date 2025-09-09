// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/external_call | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/external_call | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <iostream>

#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

extern "C" {
void hello() { std::cout << "Hello!\n"; }
}

int main() {
  proteus::init();

  auto J = proteus::JitModule();
  auto &F = J.addFunction<void>("ExternalCall");
  F.beginFunction();
  { F.call<void>("hello"); }
  F.ret();
  F.endFunction();

  J.print();
  J.compile();

  F();

  proteus::finalize();
  return 0;
}

// clang-format off
//CHECK: Hello!
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
