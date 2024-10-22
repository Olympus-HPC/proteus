// RUN: ./types.%ext | FileCheck %s --check-prefixes=CHECK

#include <cstdio>
#include <cstdlib>

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

template<typename T>
__global__ __attribute__ ((annotate("jit", 1)))
void kernel(T arg) {
  volatile T local;
  local = arg;
}

int main(int argc, char **argv) {
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
#if ENABLE_HIP
  kernel<<<1, 1>>>(1.0l);
#elif ENABLE_CUDA
  // CUDA AOT compilation with a `long double` breaks on lassen.
  // We re-test the `double` type with a different value (to avoid caching) to
  // re-use the lit CHECKs below.
  kernel<<<1, 1>>>(2.0);
#endif
  gpuErrCheck(gpuDeviceSynchronize());
  kernel<<<1, 1>>>(true);
  gpuErrCheck(gpuDeviceSynchronize());
  kernel<<<1, 1>>>('a');
  gpuErrCheck(gpuDeviceSynchronize());
  kernel<<<1, 1>>>((unsigned char) 'a');
  gpuErrCheck(gpuDeviceSynchronize());
}

// CHECK: JitCache hits 0 total 12
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0