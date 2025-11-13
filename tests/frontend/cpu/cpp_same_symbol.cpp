// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/cpp_same_symbol | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/cpp_same_symbol | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

// Tests that kernels with the same symbol from different modules are correctly
// cached and execute individually.

#include "proteus/CppJitModule.hpp"

using namespace proteus;

int main() {
  const char *Code = R"cpp(
    #include <cstdio>

    extern "C" void foo(int a) {
        printf("foo %d\n", a);
    }
   )cpp";

  CppJitModule CJM{"host", Code};
  auto Foo = CJM.getFunction<void(int)>("foo");
  Foo.run(42);

  const char *Code2 = R"cpp(
    #include <cstdio>

    extern "C" void foo(int a) {
        printf("Other foo %d\n", a);
    }
   )cpp";

  CppJitModule CJM2("host", Code2);
  auto Foo2 = CJM2.getFunction<void(int)>("foo");
  Foo2.run(42);
}

// clang-format off
// CHECK: foo 42
// CHECK: Other foo 42
// CHECK: [proteus][DispatcherHost] MemoryCache procuid 0 hits 0 accesses 2
// CHECK: [proteus][DispatcherHost] MemoryCache procuid 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: [proteus][DispatcherHost] MemoryCache procuid 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: [proteus][DispatcherHost] StorageCache procuid 0 hits 0 accesses 2
// CHECK-SECOND: [proteus][DispatcherHost] StorageCache procuid 0 hits 2 accesses 2
