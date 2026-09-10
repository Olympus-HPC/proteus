// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="cache-stats" %build/kernel_dynamic_shmem.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="cache-stats" %build/kernel_dynamic_shmem.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdint>
#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

__global__ __attribute__((annotate("jit"))) void kernel(long long *Out,
                                                        int NumElements) {
  extern __shared__ int Shmem[];

  int Tid = blockIdx.x * blockDim.x + threadIdx.x;
  int Stride = gridDim.x * blockDim.x;
  for (int I = Tid; I < NumElements; I += Stride)
    Shmem[I] = I;
  __syncthreads();

  if (Tid == 0) {
    long long Sum = 0;
    for (int I = 0; I < NumElements; ++I)
      Sum += Shmem[I];
    *Out = Sum;
  }
}

static void launch(long long *Out, uint64_t ShmemSize) {
  int NumElements = static_cast<int>(ShmemSize / sizeof(int));
  *Out = 0;

  kernel<<<1, 256, ShmemSize>>>(Out, NumElements);
  gpuErrCheck(gpuDeviceSynchronize());

  printf("Shmem %llu bytes, elements %d, sum %lld\n",
         static_cast<unsigned long long>(ShmemSize), NumElements, *Out);
}

int main() {
  long long *Out;
  gpuErrCheck(gpuMallocManaged(&Out, sizeof(long long)));

  // Exceed the 48KB default, raise the limit again, then reuse it.
  launch(Out, 64 * 1024);
  launch(Out, 128 * 1024);
  launch(Out, 64 * 1024);

  gpuErrCheck(gpuFree(Out));

  return 0;
}

// clang-format off
// CHECK: Shmem 65536 bytes, elements 16384, sum 134209536
// CHECK: Shmem 131072 bytes, elements 32768, sum 536854528
// CHECK: Shmem 65536 bytes, elements 16384, sum 134209536
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 hits 2 accesses 3
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 3 NumHits 2
// CHECK-FIRST: [proteus][JitEngineDevice] StorageCache rank 0 hits 0 accesses 1
// CHECK-SECOND: [proteus][JitEngineDevice] StorageCache rank 0 hits 1 accesses 1
// clang-format on
