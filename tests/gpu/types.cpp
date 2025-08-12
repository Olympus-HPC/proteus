// clang-format off
// RUN: rm -rf .proteus
// RUN: PROTEUS_TRACE_OUTPUT=1 ./types.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./types.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf .proteus
// clang-format on

#include <cstdio>
#include <cstdlib>

#include "gpu_common.h"
#include <proteus/JitInterface.hpp>

template <typename T>
__global__ __attribute__((annotate("jit", 1))) void kernel(T Arg) {
  volatile T Local;
  Local = Arg;
}

int main() {
  proteus::init();

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
// CUDA AOT compilation with a `long double` breaks on lassen.
// We re-test the `double` type with a different value (to avoid caching).
#if PROTEUS_ENABLE_HIP
  kernel<<<1, 1>>>(2.0l);
#elif PROTEUS_ENABLE_CUDA
  kernel<<<1, 1>>>(2.0);
#else
#error "Expected PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA"
#endif
  gpuErrCheck(gpuDeviceSynchronize());
  kernel<<<1, 1>>>(true);
  gpuErrCheck(gpuDeviceSynchronize());
  kernel<<<1, 1>>>('a');
  gpuErrCheck(gpuDeviceSynchronize());
  kernel<<<1, 1>>>((unsigned char)'a');
  gpuErrCheck(gpuDeviceSynchronize());
}

// clang-format off
// CHECK-FIRST: [ArgSpec] Replaced Function _Z6kernelIiEvT_ ArgNo 0 with value i32 1
// CHECK-FIRST: [LaunchBoundSpec] BlockSize 1
// CHECK-FIRST: [ArgSpec] Replaced Function _Z6kernelIlEvT_ ArgNo 0 with value i64 1
// CHECK-FIRST: [LaunchBoundSpec] BlockSize 1
// CHECK-FIRST: [ArgSpec] Replaced Function _Z6kernelIjEvT_ ArgNo 0 with value i32 1
// CHECK-FIRST: [LaunchBoundSpec] BlockSize 1
// CHECK-FIRST: [ArgSpec] Replaced Function _Z6kernelImEvT_ ArgNo 0 with value i64 1
// CHECK-FIRST: [LaunchBoundSpec] BlockSize 1
// CHECK-FIRST: [ArgSpec] Replaced Function _Z6kernelIxEvT_ ArgNo 0 with value i64 1
// CHECK-FIRST: [LaunchBoundSpec] BlockSize 1
// CHECK-FIRST: [ArgSpec] Replaced Function _Z6kernelIyEvT_ ArgNo 0 with value i64 1
// CHECK-FIRST: [LaunchBoundSpec] BlockSize 1
// CHECK-FIRST: [ArgSpec] Replaced Function _Z6kernelIfEvT_ ArgNo 0 with value float 1.000000e+00
// CHECK-FIRST: [LaunchBoundSpec] BlockSize 1
// CHECK-FIRST: [ArgSpec] Replaced Function _Z6kernelIdEvT_ ArgNo 0 with value double 1.000000e+00
// CHECK-FIRST: [LaunchBoundSpec] BlockSize 1
// HIP sees long double, CUDA sees double, thus the regex.
// CHECK-FIRST: [ArgSpec] Replaced Function {{_Z6kernelI[ed]EvT_}} ArgNo 0 with value double 2.000000e+00
// CHECK-FIRST: [LaunchBoundSpec] BlockSize 1
// CHECK-FIRST: [ArgSpec] Replaced Function _Z6kernelIbEvT_ ArgNo 0 with value i1 true
// CHECK-FIRST: [LaunchBoundSpec] BlockSize 1
// CHECK-FIRST: [ArgSpec] Replaced Function _Z6kernelIcEvT_ ArgNo 0 with value i8 97
// CHECK-FIRST: [LaunchBoundSpec] BlockSize 1
// CHECK-FIRST: [ArgSpec] Replaced Function _Z6kernelIhEvT_ ArgNo 0 with value i8 97
// CHECK-FIRST: [LaunchBoundSpec] BlockSize 1
// CHECK-FIRST: JitCache hits 0 total 12
// CHECK-COUNT-12: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-SECOND: JitStorageCache hits 12 total 12
