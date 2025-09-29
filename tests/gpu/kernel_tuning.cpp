// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_TUNED_KERNELS=%S/tuned_config.json PROTEUS_TRACE_OUTPUT=1 %build/kernel_tuning.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-%device_lang
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <climits>
#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.hpp>

extern "C" {
__global__ __attribute__((annotate("jit"))) void foo() {
  printf("Hello from foo\n");
}

__global__ __attribute__((annotate("jit"))) void bar() {
  printf("Hello from bar\n");
}

__global__ __attribute__((annotate("jit"))) void baz() {
  printf("Hello from baz\n");
}
}

int main() {
  proteus::init();

  foo<<<1, 1>>>();
  gpuErrCheck(gpuDeviceSynchronize());
  bar<<<1, 1>>>();
  gpuErrCheck(gpuDeviceSynchronize());
  baz<<<1, 1>>>();
  gpuErrCheck(gpuDeviceSynchronize());

  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK-CUDA: [KernelConfig] ID:foo Pipeline:default<O3>,globaldce CG:RTC SA:1 LB:0 SD:0 SDA:0 OL:3 CGL:2
// CHECK-HIP: [KernelConfig] ID:foo Pipeline:default<O3>,globaldce CG:Serial SA:1 LB:0 SD:0 SDA:0 OL:3 CGL:2
// CHECK: [CustomPipeline] default<O3>,globaldce
// CHECK: Hello from foo
// CHECK: [KernelConfig] ID:bar CG:RTC SA:1 LB:1 SD:1 SDA:1 OL:3 CGL:2
// CHECK: [LaunchBoundSpec] BlockSize 1
// CHECK: Hello from bar
// CHECK-CUDA: [KernelConfig] ID:baz CG:RTC SA:1 LB:1 SD:1 SDA:0 OL:3 CGL:3
// CHECK-HIP: [KernelConfig] ID:baz CG:{{RTC|Serial}} SA:1 LB:1 SD:1 SDA:1 OL:3 CGL:3
// CHECK: [LaunchBoundSpec] BlockSize 1
// CHECK: Hello from baz
// CHECK: JitCache hits 0 total 3
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: JitStorageCache hits 0 total 3

