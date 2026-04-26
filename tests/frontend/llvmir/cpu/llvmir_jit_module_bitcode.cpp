// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: %llvm_as %S/llvmir_jit_module_bitcode_input.ll -o %t.module.bc
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/llvmir_jit_module_bitcode %t.module.bc | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/llvmir_jit_module_bitcode %t.module.bc | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cassert>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>

#include <proteus/LLVMIRJitModule.h>

using namespace proteus;

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr, "Expected a single bitcode input path\n");
    return 1;
  }

  std::ifstream Input(argv[1], std::ios::binary);
  assert(Input && "Failed to open bitcode file");

  std::string Code((std::istreambuf_iterator<char>(Input)),
                   std::istreambuf_iterator<char>());

  LLVMIRJitModule M("host", Code, LLVMIRInputKind::Bitcode);
  auto Add = M.getFunction<int(int, int)>("add");

  const int Result = Add.run(40, 2);
  std::printf("llvmir_jit_module_bitcode: Result=%d\n", Result);
  assert(Result == 42 &&
         "LLVMIRJitModule bitcode execution returned wrong result");

  return 0;
}

// clang-format off
// CHECK: llvmir_jit_module_bitcode: Result=42
// CHECK-FIRST: [proteus][DispatcherHost] StorageCache rank 0 hits 0 accesses 1
// CHECK-SECOND: [proteus][DispatcherHost] StorageCache rank 0 hits 1 accesses 1
// clang-format on
