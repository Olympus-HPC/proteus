// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/stencil.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/stencil.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstddef>
#include <cstdio>
#include <cstdlib>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

__device__ int getGlobalThreadIdX() {
  return blockIdx.x * blockDim.x + threadIdx.x;
}
#define NWeights 5

__device__ int getThreadIdX() { return threadIdx.x; }

__attribute__((annotate("jit", 2, 3))) __global__ void
stencil1d(float *Out, float *In, int Radius, float *Weights) {
  proteus::jit_array(Weights, NWeights);
  float *Tile = proteus::shared_array<float, 10>(68);
  int Gid = getGlobalThreadIdX();
  int Tid = getThreadIdX();

  Tile[Tid + Radius] = In[Gid];
  if (Tid < Radius) {
    Tile[Tid] = In[Gid - Radius];
    Tile[Tid + blockDim.x + Radius] = In[Gid + blockDim.x];
  }
  __syncthreads();

  float Sum = 0.0f;
  for (int J = -Radius; J <= Radius; J++)
    Sum += Tile[Tid + Radius + J] * Weights[Radius + J];
  Out[Gid] = Sum;
}

int main() {
  const size_t N = 256;
  const int Radius = 2;
  const int BlockSize = 64;
  const int NumWeights = 2 * Radius + 1;

  float *In, *Out, *Weights;
  gpuErrCheck(gpuMallocManaged(&In, sizeof(float) * N));
  gpuErrCheck(gpuMallocManaged(&Out, sizeof(float) * N));
  gpuErrCheck(gpuMallocManaged(&Weights, sizeof(float) * NumWeights));

  // Initialize input array.
  for (size_t I = 0; I < N; I++) {
    In[I] = 1.0f;
  }

  // Initialize weights for a simple averaging stencil.
  for (int I = 0; I < NumWeights; I++) {
    Weights[I] = 1.0f / NumWeights;
  }

  // Calculate shared memory size: BlockSize + 2*Radius elements.
  size_t SharedMemSize = (BlockSize + 2 * Radius) * sizeof(float);
  int NumBlocks = (N + BlockSize - 1) / BlockSize;

#if PROTEUS_ENABLE_HIP
  hipLaunchKernelGGL(stencil1d, dim3(NumBlocks), dim3(BlockSize), SharedMemSize,
                     0, Out, In, Radius, Weights);
#elif PROTEUS_ENABLE_CUDA
  stencil1d<<<NumBlocks, BlockSize, SharedMemSize>>>(Out, In, Radius, Weights);
#endif
  gpuErrCheck(gpuDeviceSynchronize());

  // Verify results. With all 1s input and averaging weights, output should be
  // ~1.0.
  bool Passed = true;
  for (size_t I = Radius; I < N - Radius; I++) {
    if (Out[I] < 0.99f || Out[I] > 1.01f) {
      printf("FAIL: Out[%zu] = %f, expected ~1.0\n", I, Out[I]);
      Passed = false;
      break;
    }
  }

  if (Passed) {
    printf("PASSED\n");
  }

  gpuErrCheck(gpuFree(In));
  gpuErrCheck(gpuFree(Out));
  gpuErrCheck(gpuFree(Weights));

  return Passed ? 0 : 1;
}

// clang-format off
// CHECK-FIRST: [ArgSpec] Replaced Function {{.*}}stencil1d{{.*}} ArgNo 2 with value i32 2
// CHECK: PASSED
// CHECK-FIRST: [proteus][JitEngineDevice] StorageCache rank 0 hits 0 accesses 1
// CHECK-SECOND: [proteus][JitEngineDevice] StorageCache rank 0 hits 1 accesses 1
// clang-format on
