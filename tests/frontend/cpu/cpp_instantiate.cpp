// clang-format off
// RUN: rm -rf .proteus
// RUN: ./cpp_instantiate | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./cpp_instantiate | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf .proteus
// clang-format on

#include <iostream>

#include "proteus/CppJitModule.hpp"

using namespace proteus;

int main() {
  const char *Code = R"cpp(
    #include <cstdio>
    #include <typeinfo>

    template<int V>
    double foo(double A) {
        printf("foo V %d A %lf\n", V, A);
        return (A+1);
    }

    template<typename T>
    void bar() {
        printf("bar type %s\n", typeid(T).name());
    }
   )cpp";

  CppJitModule CJM{"host", Code};
  auto InstValue = CJM.instantiate("foo", "3");

  double Ret = InstValue.run<double>(42);
  std::cout << "Ret " << Ret << "\n";

  Ret = InstValue.run<double>(23);
  std::cout << "Ret " << Ret << "\n";

  auto InstTypeD = CJM.instantiate("bar", "double");
  auto InstTypeF = CJM.instantiate("bar", "float");
  InstTypeD.run<void>();
  InstTypeF.run<void>();

  return 0;
}

// clang-format off
// CHECK: foo V 3 A 42.000000
// CHECK: Ret 43
// CHECK: foo V 3 A 23.000000
// CHECK: Ret 24
// CHECK: bar type d
// CHECK: bar type f
// CHECK-FIRST: JitStorageCache hits 0 total 3
// CHECK-SECOND: JitStorageCache hits 3 total 3
