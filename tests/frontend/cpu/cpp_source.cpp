// clang-format off
// RUN: rm -rf .proteus
// RUN: ./cpp_source | %FILECHECK %s --check-prefixes=CHECK
// RUN: rm -rf .proteus
// clang-format on

#include <iostream>

#include "proteus/CppJitModule.hpp"

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
   )cpp";

  int V = 42;
  CppJitModule CJM{"host", Code};
  CJM.run<void>("foo_byval", V);
  CJM.run<void(int &)>("foo_byref", V);

  std::cout << "main V " << V << "\n";

  return 0;
}

// clang-format off
// CHECK: foo_byval 42
// CHECK: foo V 42
// CHECK: foo_byref V 43
// CHECK: foo V 43
// CHECK: main V 43
