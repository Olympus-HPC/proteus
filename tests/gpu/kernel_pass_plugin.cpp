// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_CODEGEN=serial PROTEUS_TRACE_OUTPUT="specialization;cache-stats" %build/kernel_pass_plugin.%ext legacy | %FILECHECK %s --check-prefixes=CHECK,LEGACY
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_CODEGEN=serial PROTEUS_TRACE_OUTPUT="specialization;cache-stats" %build/kernel_pass_plugin.%ext prepend | %FILECHECK %s --check-prefixes=CHECK,PREPEND
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_CODEGEN=serial PROTEUS_TRACE_OUTPUT="specialization;cache-stats" PROTEUS_OPT_PIPELINE="default<O3>,jit-test-pass" %build/kernel_pass_plugin.%ext load-only | %FILECHECK %s --check-prefixes=CHECK,LOAD-ONLY
// RUN: rm -rf "%t.$$.proteus"
// RUN: if [ "%device_lang" = "HIP" ]; then PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_CODEGEN=parallel PROTEUS_TRACE_OUTPUT="specialization;cache-stats" PROTEUS_OPT_PIPELINE="default<O3>,jit-test-pass" %build/kernel_pass_plugin.%ext load-only | %FILECHECK %s --check-prefixes=CHECK,PARALLEL; fi
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>
#include <string>

#include "gpu_common.h"
#include <proteus/Init.h>
#include <proteus/JitInterface.h>

__global__ __attribute__((annotate("jit"))) void kernel_pass_plugin() {
  printf("KernelPassPlugin\n");
}

int main(int argc, char **argv) {
  const std::string Mode = argc > 1 ? argv[1] : "";
  if (Mode == "legacy") {
    proteus::registerJITPassPlugin(PROTEUS_TEST_JIT_PASS_PLUGIN_PATH,
                                   "jit-test-pass");
  } else if (Mode == "prepend") {
    proteus::registerJITPassPlugin(PROTEUS_TEST_JIT_PASS_PLUGIN_PATH,
                                   "jit-test-pass",
                                   proteus::JITPassPluginPosition::Prepend);
  } else if (Mode == "load-only") {
    proteus::registerJITPassPlugin(PROTEUS_TEST_JIT_PASS_PLUGIN_PATH);
  } else {
    return 1;
  }

  kernel_pass_plugin<<<1, 1>>>();
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// LEGACY: [JITTestPluginInfo]
// LEGACY: [JITTestPass] jit-test-pass
// LEGACY: [CustomPipeline] default<O3>,jit-test-pass

// PREPEND: [JITTestPluginInfo]
// PREPEND: [JITTestPass] jit-test-pass
// PREPEND: [CustomPipeline] jit-test-pass,default<O3>

// LOAD-ONLY: [JITTestPluginInfo]
// LOAD-ONLY: [JITTestPass] jit-test-pass
// LOAD-ONLY-NOT: [JITTestPass] jit-test-pass
// LOAD-ONLY: [CustomPipeline] default<O3>,jit-test-pass

// PARALLEL: [JITTestPluginInfo]
// PARALLEL: [JITTestPass] jit-test-pass

// CHECK: KernelPassPlugin
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 hits 0 accesses 1
