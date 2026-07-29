// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_CODEGEN=serial PROTEUS_TRACE_OUTPUT="specialization;cache-stats" PROTEUS_OPT_PIPELINE="default<O3>,jit-test-pass" %build/jit_pass_plugin_cmake_load_only | %FILECHECK %s
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <iostream>

#include <proteus/JitInterface.h>

__attribute__((annotate("jit"))) int add_four(int x) {
  proteus::jit_arg(x);
  return x + 4;
}

int main() {
  std::cout << add_four(7) << "\n";
  return 0;
}

// CHECK: [JITTestPluginInfo]
// CHECK-NOT: [JITTestPluginInfo]
// CHECK: [JITTestPass] jit-test-pass
// CHECK-NOT: [JITTestPass] jit-test-pass
// CHECK: [CustomPipeline] default<O3>,jit-test-pass
// CHECK: 11
// CHECK: [proteus][JitEngineHost] MemoryCache rank 0 hits 0 accesses 1
