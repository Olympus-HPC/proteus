// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/types.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/types.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>
#include <cstdlib>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

template <typename T>
__global__ __attribute__((annotate("jit", 1))) void kernel(T Arg) {
  volatile T Local;
  Local = Arg;
}

int main() {
  kernel<<<1, 1>>>(1);
  gpuErrCheck(gpuDeviceSynchronize());
  kernel<<<1, 1>>>(1l);
  gpuErrCheck(gpuDeviceSynchronize());
  kernel<<<1, 1>>>(1u);
  gpuErrCheck(gpuDeviceSynchronize());
  kernel<<<1, 1>>>(1ul);
  gpuErrCheck(gpuDeviceSynchronize());
  kernel<<<1, 1>>>(1ll);
  gpuErrCheck(gpuDeviceSynchronize());
  kernel<<<1, 1>>>(1ull);
  gpuErrCheck(gpuDeviceSynchronize());
  kernel<<<1, 1>>>(1.0f);
  gpuErrCheck(gpuDeviceSynchronize());
  kernel<<<1, 1>>>(1.0);
  gpuErrCheck(gpuDeviceSynchronize());
  // CUDA/HIP do not support long double on GPUs, demoting to double, hence we
  // do not support or test.
  kernel<<<1, 1>>>(true);
  gpuErrCheck(gpuDeviceSynchronize());
  kernel<<<1, 1>>>('a');
  gpuErrCheck(gpuDeviceSynchronize());
  kernel<<<1, 1>>>((unsigned char)'a');
  gpuErrCheck(gpuDeviceSynchronize());
}

// clang-format off
// CHECK-FIRST: [ArgSpec] Replaced Function _Z6kernelIiEvT_ ArgNo 0 with value i32 1
// CHECK-FIRST: [LaunchBoundSpec] MaxThreads 1 MinBlocksPerSM 0
// CHECK-FIRST: [ArgSpec] Replaced Function _Z6kernelIlEvT_ ArgNo 0 with value i64 1
// CHECK-FIRST: [LaunchBoundSpec] MaxThreads 1 MinBlocksPerSM 0
// CHECK-FIRST: [ArgSpec] Replaced Function _Z6kernelIjEvT_ ArgNo 0 with value i32 1
// CHECK-FIRST: [LaunchBoundSpec] MaxThreads 1 MinBlocksPerSM 0
// CHECK-FIRST: [ArgSpec] Replaced Function _Z6kernelImEvT_ ArgNo 0 with value i64 1
// CHECK-FIRST: [LaunchBoundSpec] MaxThreads 1 MinBlocksPerSM 0
// CHECK-FIRST: [ArgSpec] Replaced Function _Z6kernelIxEvT_ ArgNo 0 with value i64 1
// CHECK-FIRST: [LaunchBoundSpec] MaxThreads 1 MinBlocksPerSM 0
// CHECK-FIRST: [ArgSpec] Replaced Function _Z6kernelIyEvT_ ArgNo 0 with value i64 1
// CHECK-FIRST: [LaunchBoundSpec] MaxThreads 1 MinBlocksPerSM 0
// CHECK-FIRST: [ArgSpec] Replaced Function _Z6kernelIfEvT_ ArgNo 0 with value float 1.000000e+00
// CHECK-FIRST: [LaunchBoundSpec] MaxThreads 1 MinBlocksPerSM 0
// CHECK-FIRST: [ArgSpec] Replaced Function _Z6kernelIdEvT_ ArgNo 0 with value double 1.000000e+00
// CHECK-FIRST: [LaunchBoundSpec] MaxThreads 1 MinBlocksPerSM 0
// CHECK-FIRST: [ArgSpec] Replaced Function _Z6kernelIbEvT_ ArgNo 0 with value i1 true
// CHECK-FIRST: [LaunchBoundSpec] MaxThreads 1 MinBlocksPerSM 0
// CHECK-FIRST: [ArgSpec] Replaced Function _Z6kernelIcEvT_ ArgNo 0 with value i8 97
// CHECK-FIRST: [LaunchBoundSpec] MaxThreads 1 MinBlocksPerSM 0
// CHECK-FIRST: [ArgSpec] Replaced Function _Z6kernelIhEvT_ ArgNo 0 with value i8 97
// CHECK-FIRST: [LaunchBoundSpec] MaxThreads 1 MinBlocksPerSM 0
// CHECK-FIRST: [proteus][JitEngineDevice] MemoryCache rank 0 hits 0 accesses 11
// CHECK-COUNT-11: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-SECOND: [proteus][JitEngineDevice] StorageCache rank 0 hits 11 accesses 11
