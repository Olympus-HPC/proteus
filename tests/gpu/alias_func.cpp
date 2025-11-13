// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/alias_func.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/alias_func.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>

#include "gpu_common.h"

#if defined(__HIP_DEVICE_COMPILE__) ||                                         \
    (defined(__CUDA__) && defined(__CUDA_ARCH__))
// 1) Forward‐declare the real device function
extern "C" __device__ __attribute__((used)) void foo(int V);

// 2) Create the alias (must refer to the mangled/assembler name)
extern "C" __device__ __attribute__((used)) void foo_alias(int V)
    __attribute__((alias("foo")));

// 3) Define the aliasee
extern "C" __device__ __attribute__((used)) void foo(int V) {
  printf("foo %d\n", V);
}
#endif

// A trivial kernel that forces emission of both symbols
__global__ __attribute__((annotate("jit"))) void kernel() {
#if defined(__HIP_DEVICE_COMPILE__) ||                                         \
    (defined(__CUDA__) && defined(__CUDA_ARCH__))
  printf("Kernel\n");
  foo_alias(42);
#endif
}

int main() {
  kernel<<<1, 1>>>();
  gpuErrCheck(gpuDeviceSynchronize());
  return 0;
}

// clang-format off
// CHECK: Kernel
// CHECK: foo 42
// CHECK: [proteus][JitEngineDevice] MemoryCache procuid 0 hits 0 accesses 1
// CHECK: [proteus][JitEngineDevice] MemoryCache procuid 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: [proteus][JitEngineDevice] StorageCache procuid 0 hits 0 accesses 1
// CHECK-SECOND: [proteus][JitEngineDevice] StorageCache procuid 0 hits 1 accesses 1
