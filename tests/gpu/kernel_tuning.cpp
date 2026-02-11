// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_TUNED_KERNELS=%S/tuned_config.json PROTEUS_TRACE_OUTPUT=1 %build/kernel_tuning.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-%device_lang
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <climits>
#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

extern "C" {
__global__ __attribute__((annotate("jit"))) void foo() {
  printf("Hello from foo\n");
}

__global__ __attribute__((annotate("jit"))) void bar() {
  unsigned int Idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (Idx == 0)
    printf("Hello from bar\n");
}

__global__ __attribute__((annotate("jit"))) void baz() {
  printf("Hello from baz\n");
}
}

int main() {
  foo<<<1, 1>>>();
  gpuErrCheck(gpuDeviceSynchronize());
  bar<<<1, 1>>>();
  gpuErrCheck(gpuDeviceSynchronize());
  bar<<<1, 4>>>();
  gpuErrCheck(gpuDeviceSynchronize());
  baz<<<1, 1>>>();
  gpuErrCheck(gpuDeviceSynchronize());

  return 0;
}

// clang-format off
// CHECK-CUDA: [KernelConfig] ID:foo Pipeline:default<O3>,globaldce CG:RTC SA:1 LB:0 SD:0 SDR:0 OL:3 CGL:2 TMT:1 BPSM:4
// CHECK-HIP: [KernelConfig] ID:foo Pipeline:default<O3>,globaldce CG:Serial SA:1 LB:0 SD:0 SDR:0 OL:3 CGL:2 TMT:1 BPSM:4
// CHECK: [CustomPipeline] default<O3>,globaldce
// CHECK: Hello from foo
// CHECK: [KernelConfig] ID:bar CG:RTC SA:1 LB:1 SD:1 SDR:1 OL:3 CGL:2 TMT:4 BPSM:4
// CHECK: [LaunchBoundSpec] MaxThreads 1 MinBlocksPerSM 0
// CHECK: Hello from bar
// CHECK: [KernelConfig] ID:bar CG:RTC SA:1 LB:1 SD:1 SDR:1 OL:3 CGL:2 TMT:4 BPSM:4
// CHECK: [LaunchBoundSpec] MaxThreads 4 MinBlocksPerSM 4
// CHECK: Hello from bar
// CHECK-CUDA: [KernelConfig] ID:baz CG:RTC SA:1 LB:1 SD:1 SDR:0 OL:3 CGL:3 TMT:-1 BPSM:0
// CHECK-HIP: [KernelConfig] ID:baz CG:{{RTC|Serial|Parallel}} SA:1 LB:1 SD:1 SDR:1 OL:3 CGL:3 TMT:-1 BPSM:0
// CHECK: [LaunchBoundSpec] MaxThreads 1 MinBlocksPerSM 0
// CHECK: Hello from baz
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 hits 0 accesses 4
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: [proteus][JitEngineDevice] StorageCache rank 0 hits 0 accesses 4
