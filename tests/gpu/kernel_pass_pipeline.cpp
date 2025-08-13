// clang-format off
// RUN: rm -rf .proteus
// RUN: PROTEUS_CODEGEN=serial PROTEUS_TRACE_OUTPUT=1 ./kernel_pass_pipeline.%ext | %FILECHECK %s --check-prefixes=CHECK
// RUN: rm -rf .proteus
// RUN: PROTEUS_TRACE_OUTPUT=1 PROTEUS_CODEGEN=serial PROTEUS_TRACE_OUTPUT=1 PROTEUS_OPT_PIPELINE='default<O3>' ./kernel_pass_pipeline.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK3
// RUN: rm -rf .proteus
// RUN: PROTEUS_TRACE_OUTPUT=1 PROTEUS_CODEGEN=serial PROTEUS_TRACE_OUTPUT=1 PROTEUS_OPT_PIPELINE='default<O2>' ./kernel_pass_pipeline.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK2
// RUN: rm -rf .proteus
// RUN: PROTEUS_TRACE_OUTPUT=1 PROTEUS_CODEGEN=serial PROTEUS_TRACE_OUTPUT=1 PROTEUS_OPT_PIPELINE='default<O1>' ./kernel_pass_pipeline.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK1
// RUN: rm -rf .proteus
// RUN: PROTEUS_TRACE_OUTPUT=1 PROTEUS_CODEGEN=serial PROTEUS_TRACE_OUTPUT=1 PROTEUS_OPT_PIPELINE='default<Os>' ./kernel_pass_pipeline.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECKs
// RUN: rm -rf .proteus
// RUN: PROTEUS_TRACE_OUTPUT=1 PROTEUS_CODEGEN=serial PROTEUS_TRACE_OUTPUT=1 PROTEUS_OPT_PIPELINE='default<Oz>' ./kernel_pass_pipeline.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECKz
// RUN: rm -rf .proteus
// clang-format on
#include <climits>
#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.hpp>

__global__ __attribute__((annotate("jit"))) void kernel() {
  printf("Kernel\n");
}

int main() {
  proteus::init();

  kernel<<<1, 1>>>();
  gpuErrCheck(gpuDeviceSynchronize());

  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK: [LaunchBoundSpec] GridSize 1 BlockSize 1
// CHECK1: [CustomPipeline] default<O1>
// CHECK2: [CustomPipeline] default<O2>
// CHECK3: [CustomPipeline] default<O3>
// CHECKs: [CustomPipeline] default<Os>
// CHECKz: [CustomPipeline] default<Oz>
// CHECK: Kernel
// CHECK: JitCache hits 0 total 1
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
