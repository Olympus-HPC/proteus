// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/cpp_store_handle | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/cpp_store_handle | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include "proteus/CppJitModule.h"

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

  return 0;
}

// clang-format off
// CHECK: Function 42
// CHECK: Function 42
// CHECK-FIRST: [proteus][DispatcherHost] StorageCache rank 0 hits 0 accesses 1
// CHECK-SECOND: [proteus][DispatcherHost] StorageCache rank 0 hits 1 accesses 1
