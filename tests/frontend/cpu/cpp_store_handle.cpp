// clang-format off
// RUN: rm -rf .proteus
// RUN: ./cpp_store_handle | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./cpp_store_handle | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf .proteus
// clang-format on

#include "proteus/CppJitModule.hpp"

using namespace proteus;

struct FuncStore {
  CppJitModule::FunctionHandle<void(int)> Func;
};

void launcher(FuncStore &FS) { FS.Func.run(42); }

int main() {
  const char *Code = R"cpp(
    #include <cstdio>

    extern "C" void foo(int a) {
        printf("Function %d\n", a);
    }
   )cpp";

  CppJitModule CJM{"host", Code};
  auto Func = CJM.getFunction<void(int)>("foo");
  Func.run(42);
  FuncStore FS{Func};
  launcher(FS);
}

// clang-format off
// CHECK: Function 42
// CHECK: Function 42
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
