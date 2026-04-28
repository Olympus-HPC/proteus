// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/llvmir_jit_module_source | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/llvmir_jit_module_source | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cassert>
#include <cstdio>

#include <proteus/LLVMIRJitModule.h>

using namespace proteus;

int main() {
  static constexpr const char *Code = R"llvm(
define i32 @add(i32 %a, i32 %b) {
entry:
  %sum = add i32 %a, %b
  ret i32 %sum
}
)llvm";

  LLVMIRJitModule M("host", Code);
  auto Add = M.getFunction<int(int, int)>("add");

  const int Result = Add.run(40, 2);
  std::printf("llvmir_jit_module_source: Result=%d\n", Result);
  assert(Result == 42 &&
         "LLVMIRJitModule host execution returned wrong result");

  return 0;
}

// clang-format off
// CHECK: llvmir_jit_module_source: Result=42
// CHECK-FIRST: [proteus][DispatcherHost] StorageCache rank 0 hits 0 accesses 1
// CHECK-SECOND: [proteus][DispatcherHost] StorageCache rank 0 hits 1 accesses 1
// clang-format on
