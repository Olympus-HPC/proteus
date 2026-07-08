// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_CODEGEN=serial PROTEUS_TRACE_OUTPUT="specialization;cache-stats" %build/jit_pass_plugin_cmake | %FILECHECK %s
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <iostream>

#include <proteus/JitInterface.h>

__attribute__((annotate("jit"))) int add_two(int x) {
  proteus::jit_arg(x);
  return x + 2;
}

int main() {
  std::cout << add_two(5) << "\n";
  return 0;
}

// CHECK: [JITTestPass]
// CHECK: [CustomPipeline] default<O3>,jit-test-pass
// CHECK: 7
// CHECK: [proteus][JitEngineHost] MemoryCache rank 0 hits 0 accesses 1
