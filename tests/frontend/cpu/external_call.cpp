// RUN: rm -rf .proteus
// RUN: ./external_call | %FILECHECK %s --check-prefixes=CHECK
// RUN: rm -rf .proteus

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
