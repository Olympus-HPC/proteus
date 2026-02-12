// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/cpp_source | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/cpp_source | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <iostream>

#include "proteus/CppJitModule.h"

using namespace proteus;

int main() {
  const char *Code = R"cpp(
    #include <cstdio>
    template<typename T>
    T foo(int V) {
        printf("foo V %d\n", V);
        return 0;
    }

    extern "C" void foo_byval(int V) {
      printf("foo_byval %d\n", V);
      foo<int>(V);
    }

    extern "C" void foo_byref(int &V) {
      V++;
      printf("foo_byref V %d\n", V);
      foo<int>(V);
    }

    extern "C" void bar() {
      printf("bar from code\n");
    }
   )cpp";

  int V = 42;
  CppJitModule CJM{"host", Code};
  auto FooByVal = CJM.getFunction<void(int)>("foo_byval");
  auto FooByRef = CJM.getFunction<void(int &)>("foo_byref");
  FooByVal.run(V);
  FooByRef.run(V);
  auto Bar = CJM.getFunction<void()>("bar");
  Bar.run();

  const char *Code2 = R"cpp(
     #include <cstdio>
     extern "C" void bar() {
       printf("bar from code2\n");
     }
    )cpp";
  CppJitModule CJM2{"host", Code2};
  auto Bar2 = CJM2.getFunction<void()>("bar");
  Bar2.run();

  std::cout << "main V " << V << "\n";

  return 0;
}

// clang-format off
// CHECK: foo_byval 42
// CHECK: foo V 42
// CHECK: foo_byref V 43
// CHECK: foo V 43
// CHECK: main V 43
// CHECK-FIRST: [proteus][DispatcherHost] StorageCache rank 0 hits 0 accesses 2
// CHECK-SECOND: [proteus][DispatcherHost] StorageCache rank 0 hits 2 accesses 2
