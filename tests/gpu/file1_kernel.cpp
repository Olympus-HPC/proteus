// RUN: ./multi_file.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./multi_file.%ext | FileCheck %s --check-prefixes=CHECK,CHECK-SECOND
#include <climits>
#include <cstdio>

#include "gpu_common.h"

#define gpuErrCheck(CALL)                                                      \
  {                                                                            \
    gpuError_t err = CALL;                                                     \
    if (err != gpuSuccess) {                                                   \
      printf("ERROR @ %s:%d ->  %s\n", __FILE__, __LINE__,                     \
             gpuGetErrorString(err));                                          \
      abort();                                                                 \
    }                                                                          \
  }

__global__ __attribute__((annotate("jit"))) static void kernel() {
  printf("File1 Kernel\n");
}

void foo();
int main() {
  kernel<<<1, 1>>>();
  gpuErrCheck(gpuDeviceSynchronize());
  foo();
  return 0;
}

// CHECK: File1 Kernel
// CHECK: File2 Kernel
// CHECK: JitCache hits 0 total 2
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 2
// CHECK-SECOND: JitStorageCache hits 2 total 2
