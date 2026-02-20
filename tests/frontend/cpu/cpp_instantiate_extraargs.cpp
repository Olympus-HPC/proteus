// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/cpp_instantiate_extraargs | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/cpp_instantiate_extraargs | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <iostream>

#include "proteus/CppJitModule.h"

using namespace proteus;

int main() {
  const char *Code = R"cpp(
    #ifndef MY_OFFSET
    #define MY_OFFSET 0
    #endif

    template<typename T>
    T foo(T A) {
        return A + static_cast<T>(MY_OFFSET);
    }
  )cpp";

  CppJitModule CJM{"host", Code, {"-DMY_OFFSET=10"}};
  auto Inst = CJM.instantiate("foo", "double");

  double Ret = Inst.run<double>(5.0);
  std::cout << "Ret " << Ret << "\n";

  Ret = Inst.run<double>(20.0);
  std::cout << "Ret " << Ret << "\n";

  return 0;
}

// clang-format off
// CHECK: Ret 15
// CHECK: Ret 30
// CHECK-FIRST: [proteus][DispatcherHost] StorageCache rank 0 hits 0 accesses 1
// CHECK-SECOND: [proteus][DispatcherHost] StorageCache rank 0 hits 1 accesses 1
