// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_CODEGEN=serial PROTEUS_TRACE_OUTPUT="specialization;cache-stats" %build/jit_pass_plugin | %FILECHECK %s
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <iostream>

#include <proteus/Init.h>
#include <proteus/JitInterface.h>

__attribute__((annotate("jit"))) int add_one(int x) {
  proteus::jit_arg(x);
  return x + 1;
}

int main() {
  proteus::registerJITPassPlugin(PROTEUS_TEST_JIT_PASS_PLUGIN_PATH,
                                 "jit-test-pass");
  std::cout << add_one(4) << "\n";
  return 0;
}

// CHECK: [JITTestPass]
// CHECK: [CustomPipeline] default<O3>,jit-test-pass
// CHECK: 5
// CHECK: [proteus][JitEngineHost] MemoryCache rank 0 hits 0 accesses 1
