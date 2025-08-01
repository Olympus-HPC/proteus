// clang-format off
// RUN: rm -rf .proteus
// RUN: PROTEUS_TRACE_OUTPUT=1 ./kernel_args_annot_mix.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./kernel_args_annot_mix.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf .proteus
// clang-format on
#include <climits>
#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.hpp>

__global__
    __attribute__((annotate("jit", R"({"arg":1})", 2, R"({"arg":3})"))) void
    kernel(int Arg1, int Arg2, int Arg3) {
  printf("Kernel arg %d\n", Arg1 + Arg2 + Arg3);
}

int main() {
  proteus::init();

  kernel<<<1, 1>>>(3, 2, 1);
  gpuErrCheck(gpuDeviceSynchronize());

  proteus::finalize();
  return 0;
}
// clang-format off
// CHECK-FIRST: [ArgSpec] Replaced Function _Z6kerneliii ArgNo 0 with value i32 3
// CHECK-FIRST: [ArgSpec] Replaced Function _Z6kerneliii ArgNo 1 with value i32 2
// CHECK-FIRST: [ArgSpec] Replaced Function _Z6kerneliii ArgNo 2 with value i32 1
// CHECK-FIRST: [LaunchBoundSpec] GridSize 1 BlockSize 1
// CHECK: Kernel arg 6
// CHECK: JitCache hits 0 total 1
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
