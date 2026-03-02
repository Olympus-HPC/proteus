// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/mlir_host_jit_execute | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/mlir_host_jit_execute | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cassert>
#include <cstddef>

#include <proteus/JitFrontend.h>

using namespace proteus;

int main() {
  JitModule J("host", "mlir");
  auto &Add = J.addFunction<size_t(size_t, size_t)>("add");

  Add.beginFunction();
  {
    auto [A, B] = Add.getArgs();
    Add.ret(A + B);
  }
  Add.endFunction();

  J.compile();

  const size_t Result = Add(40, 2);
  printf("result: %zu\n", Result);
  assert(Result == 42 && "MLIR host JIT execution returned wrong result");

  return 0;
}

// clang-format off
// CHECK: result: 42
// CHECK-FIRST: [proteus][DispatcherHost] StorageCache rank 0 hits 0 accesses 1
// CHECK-SECOND: [proteus][DispatcherHost] StorageCache rank 0 hits 1 accesses 1
// clang-format on
