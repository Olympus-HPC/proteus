// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_CODEGEN=serial PROTEUS_TRACE_OUTPUT="specialization;cache-stats" %build/kernel_pass_plugin.%ext | %FILECHECK %s
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>

#include "gpu_common.h"
#include <proteus/Init.h>
#include <proteus/JitInterface.h>

__global__ __attribute__((annotate("jit"))) void kernel_pass_plugin() {
  printf("KernelPassPlugin\n");
}

int main() {
  proteus::registerJITPassPlugin(PROTEUS_TEST_JIT_PASS_PLUGIN_PATH,
                                 "jit-test-pass");
  kernel_pass_plugin<<<1, 1>>>();
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// CHECK-DAG: [JITTestPass]
// CHECK-DAG: [CustomPipeline] default<O3>,jit-test-pass
// CHECK-DAG: KernelPassPlugin
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 hits 0 accesses 1
