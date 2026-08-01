// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_CODEGEN=serial PROTEUS_TRACE_OUTPUT="specialization;cache-stats" %build/jit_pass_plugin_position append | %FILECHECK %s --check-prefix=APPEND
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_CODEGEN=serial PROTEUS_TRACE_OUTPUT="specialization;cache-stats" %build/jit_pass_plugin_position prepend | %FILECHECK %s --check-prefix=PREPEND
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_CODEGEN=serial PROTEUS_TRACE_OUTPUT="specialization;cache-stats" %build/jit_pass_plugin_position load-only | %FILECHECK %s --check-prefix=LOAD-DEFAULT
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_CODEGEN=serial PROTEUS_TRACE_OUTPUT="specialization;cache-stats" PROTEUS_OPT_PIPELINE="default<O3>,jit-test-pass" %build/jit_pass_plugin_position load-only | %FILECHECK %s --check-prefix=LOAD-ONLY
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_CODEGEN=serial PROTEUS_TRACE_OUTPUT="specialization;cache-stats" %build/jit_pass_plugin_position order | %FILECHECK %s --check-prefix=ORDER
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <iostream>
#include <string>

#include <proteus/Init.h>
#include <proteus/JitInterface.h>

__attribute__((annotate("jit"))) int add_three(int x) {
  proteus::jit_arg(x);
  return x + 3;
}

int main(int argc, char **argv) {
  const std::string Mode = argc > 1 ? argv[1] : "";
  if (Mode == "append") {
    proteus::registerJITPassPlugin(PROTEUS_TEST_JIT_PASS_PLUGIN_PATH,
                                   "jit-test-pass",
                                   proteus::JITPassPluginPosition::Append);
  } else if (Mode == "prepend") {
    proteus::registerJITPassPlugin(PROTEUS_TEST_JIT_PASS_PLUGIN_PATH,
                                   "jit-test-pass",
                                   proteus::JITPassPluginPosition::Prepend);
  } else if (Mode == "load-only") {
    proteus::registerJITPassPlugin(PROTEUS_TEST_JIT_PASS_PLUGIN_PATH);
  } else if (Mode == "order") {
    proteus::registerJITPassPlugin(PROTEUS_TEST_JIT_PASS_PLUGIN_PATH,
                                   "jit-test-pass-prepend-first",
                                   proteus::JITPassPluginPosition::Prepend);
    proteus::registerJITPassPlugin(PROTEUS_TEST_JIT_PASS_PLUGIN_PATH,
                                   "jit-test-pass-append-first",
                                   proteus::JITPassPluginPosition::Append);
    proteus::registerJITPassPlugin(PROTEUS_TEST_JIT_PASS_PLUGIN_PATH);
    proteus::registerJITPassPlugin(PROTEUS_TEST_JIT_PASS_PLUGIN_PATH,
                                   "jit-test-pass-prepend-second",
                                   proteus::JITPassPluginPosition::Prepend);
    proteus::registerJITPassPlugin(PROTEUS_TEST_JIT_PASS_PLUGIN_PATH,
                                   "jit-test-pass-append-second",
                                   proteus::JITPassPluginPosition::Append);
  } else {
    return 1;
  }

  std::cout << add_three(6) << "\n";
  return 0;
}

// APPEND: [JITTestPluginInfo]
// APPEND-NOT: [JITTestPluginInfo]
// APPEND: [JITTestPass] jit-test-pass
// APPEND-NOT: [JITTestPass] jit-test-pass
// APPEND: [CustomPipeline] default<O3>,jit-test-pass
// APPEND: 9

// PREPEND: [JITTestPluginInfo]
// PREPEND-NOT: [JITTestPluginInfo]
// PREPEND: [JITTestPass] jit-test-pass
// PREPEND-NOT: [JITTestPass] jit-test-pass
// PREPEND: [CustomPipeline] jit-test-pass,default<O3>
// PREPEND: 9

// LOAD-DEFAULT: [JITTestPluginInfo]
// LOAD-DEFAULT-NOT: [JITTestPass]
// LOAD-DEFAULT-NOT: [CustomPipeline]
// LOAD-DEFAULT: 9

// LOAD-ONLY: [JITTestPluginInfo]
// LOAD-ONLY-NOT: [JITTestPluginInfo]
// LOAD-ONLY: [JITTestPass] jit-test-pass
// LOAD-ONLY-NOT: [JITTestPass] jit-test-pass
// LOAD-ONLY: [CustomPipeline] default<O3>,jit-test-pass
// LOAD-ONLY: 9

// ORDER: [JITTestPluginInfo]
// ORDER-NOT: [JITTestPluginInfo]
// ORDER: [JITTestPass] jit-test-pass-prepend-first
// ORDER: [JITTestPass] jit-test-pass-prepend-second
// ORDER: [JITTestPass] jit-test-pass-append-first
// ORDER: [JITTestPass] jit-test-pass-append-second
// ORDER: [CustomPipeline] jit-test-pass-prepend-first,
// ORDER-SAME: jit-test-pass-prepend-second,default<O3>,
// ORDER-SAME: jit-test-pass-append-first,jit-test-pass-append-second
// ORDER: 9
