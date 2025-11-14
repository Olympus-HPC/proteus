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
int add(int A, int B) { return A + B; }
}

int main() {
  proteus::init();

  auto J = proteus::JitModule();
  auto &F = J.addFunction<int(void)>("ExternalCall");
  F.beginFunction();
  {
    F.call<void(void)>("hello");
    auto V1 = F.defVar<int>(22);
    auto V2 = F.defVar<int>(20);
    auto V3 = F.call<int(int, int)>("add", V1, V2);
    F.ret(V3);
  }
  F.endFunction();

  J.print();
  J.compile();

  int V = F();
  std::cout << "V " << V << "\n";

  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK: Hello!
// CHECK-NEXT: V 42
// CHECK-FIRST: [proteus][DispatcherHost] StorageCache rank 0 hits 0 accesses 1
// CHECK-SECOND: [proteus][DispatcherHost] StorageCache rank 0 hits 1 accesses 1
