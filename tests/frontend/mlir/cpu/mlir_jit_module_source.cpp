// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/mlir_jit_module_source | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/mlir_jit_module_source | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cassert>
#include <cstdio>

#include <proteus/MLIRJitModule.h>

using namespace proteus;

int main() {
  static constexpr const char *Code = R"mlir(
module {
  func.func @add(%a: i32, %b: i32) -> i32 {
    %sum = arith.addi %a, %b : i32
    return %sum : i32
  }
}
)mlir";

  MLIRJitModule M("host", Code);
  auto Add = M.getFunction<int(int, int)>("add");

  const int Result = Add.run(40, 2);
  std::printf("mlir_jit_module_source: Result=%d\n", Result);
  assert(Result == 42 && "MLIRJitModule host execution returned wrong result");

  return 0;
}

// clang-format off
// CHECK: mlir_jit_module_source: Result=42
// CHECK-FIRST: [proteus][DispatcherHost] StorageCache rank 0 hits 0 accesses 1
// CHECK-SECOND: [proteus][DispatcherHost] StorageCache rank 0 hits 1 accesses 1
// clang-format on
