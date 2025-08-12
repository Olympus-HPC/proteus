// clang-format off
// RUN: rm -rf .proteus
// RUN: PROTEUS_CODEGEN=serial PROTEUS_TRACE_OUTPUT=1 ./kernel_pass_pipeline.%ext | %FILECHECK %s --check-prefixes=CHECK
// Second run uses the object cache.
// RUN: rm -rf .proteus
// RUN: PROTEUS_CODEGEN=serial PROTEUS_TRACE_OUTPUT=1 PROTEUS_OPT_PIPELINE='default<O3>' ./kernel_pass_pipeline.%ext | %FILECHECK %s --check-prefixes=CHECK
// RUN: rm -rf .proteus
// RUN: PROTEUS_CODEGEN=serial PROTEUS_TRACE_OUTPUT=1 PROTEUS_OPT_PIPELINE='default<O2>' ./kernel_pass_pipeline.%ext | %FILECHECK %s --check-prefixes=CHECK
// RUN: rm -rf .proteus
// RUN: PROTEUS_CODEGEN=serial PROTEUS_TRACE_OUTPUT=1 PROTEUS_OPT_PIPELINE='default<O1>' ./kernel_pass_pipeline.%ext | %FILECHECK %s --check-prefixes=CHECK
// RUN: rm -rf .proteus
// RUN: PROTEUS_CODEGEN=serial PROTEUS_TRACE_OUTPUT=1 PROTEUS_OPT_PIPELINE='default<Os>' ./kernel_pass_pipeline.%ext | %FILECHECK %s --check-prefixes=CHECK
// RUN: rm -rf .proteus
// RUN: PROTEUS_CODEGEN=serial PROTEUS_TRACE_OUTPUT=1 PROTEUS_OPT_PIPELINE='default<Oz>' ./kernel_pass_pipeline.%ext | %FILECHECK %s --check-prefixes=CHECK
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
// CHECK: Kernel
// CHECK: JitCache hits 0 total 1
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
