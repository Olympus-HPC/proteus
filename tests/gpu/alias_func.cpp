// clang-format off
// RUN: rm -rf .proteus
// RUN: ./alias_func.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./alias_func.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf .proteus
// clang-format on

#include <cstdio>

#include "gpu_common.h"

#if defined(__HIP_DEVICE_COMPILE__) ||                                         \
    (defined(__CUDA__) && defined(__CUDA_ARCH__))
// 1) Forward‐declare the real device function
extern "C" __device__ __attribute__((used)) void foo(void *ptr, int v);

// 2) Create the alias (must refer to the mangled/assembler name)
extern "C" __device__ __attribute__((used)) void foo_alias(void *ptr, int v)
    __attribute__((alias("foo")));

// 3) Define the aliasee
extern "C" __device__ __attribute__((used)) void foo() { printf("foo\n"); }
#endif

// A trivial kernel that forces emission of both symbols
__global__ __attribute__((annotate("jit"))) void kernel() {
#if defined(__HIP_DEVICE_COMPILE__) ||                                         \
    (defined(__CUDA__) && defined(__CUDA_ARCH__))
  printf("Kernel\n");
  foo_alias(nullptr, 42);
#endif
}

int main() {
  kernel<<<1, 1>>>();
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// CHECK: Kernel
// CHECK: foo
// CHECK: JitCache hits 0 total 1
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
