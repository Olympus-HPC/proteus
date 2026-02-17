// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization" %build/stencil_lambda.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/stencil_lambda.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
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

__device__ int getThreadIdX() { return threadIdx.x; }

#define NWeights 5
#define MaxTileSize 128

// Kernel passes weights as explicit argument to the lambda so jit_array works.
template <typename Lambda>
__global__ __attribute__((annotate("jit"))) void stencilKernel(Lambda Body,
                                                               float *Weights) {
  Body(Weights);
}

template <typename Lambda>
void launchStencil(int NumBlocks, int BlockSize, Lambda &&Body,
                   float *Weights) {
  proteus::register_lambda(Body);
  stencilKernel<<<NumBlocks, BlockSize>>>(Body, Weights);
  gpuErrCheck(gpuDeviceSynchronize());
}

int main() {
  const size_t N = 256;
  const int Radius = 2;
  const int BlockSize = 64;
  const int NumWeights = 2 * Radius + 1;
  const int TileSize = BlockSize + 2 * Radius;

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

  int NumBlocks = (N + BlockSize - 1) / BlockSize;

  auto Kernel = [
    =, Radius = proteus::jit_variable(Radius),
    TileSize = proteus::jit_variable(TileSize)
  ] __device__(float *Weights) __attribute__((annotate("jit"))) {
    proteus::jit_array(Weights, NWeights);
    float *Tile = proteus::shared_array<float, MaxTileSize>(TileSize);
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
  };
  launchStencil(NumBlocks, BlockSize, Kernel, Weights);

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
// CHECK-FIRST: [LambdaSpec] Replacing slot {{[0-9]+}} with i32 2
// CHECK-FIRST: [LambdaSpec] Replacing slot {{[0-9]+}} with i32 68
// CHECK-FIRST: [SharedArray] Replace CB float* proteus::shared_array<float, 128ul, 0>(unsigned long, unsigned long) with @.proteus.shared
// CHECK: PASSED
// CHECK-FIRST: [proteus][JitEngineDevice] StorageCache rank 0 hits 0 accesses 1
// CHECK-SECOND: [proteus][JitEngineDevice] StorageCache rank 0 hits 1 accesses 1
// clang-format on
