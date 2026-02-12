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
                                                               float *weights) {
  Body(weights);
}

template <typename Lambda>
void launchStencil(int numBlocks, int blockSize, Lambda &&Body,
                   float *weights) {
  proteus::register_lambda(Body);
  stencilKernel<<<numBlocks, blockSize>>>(Body, weights);
  gpuErrCheck(gpuDeviceSynchronize());
}

int main() {
  proteus::init();

  const size_t N = 256;
  const int radius = 2;
  const int blockSize = 64;
  const int numWeights = 2 * radius + 1;
  const int tileSize = blockSize + 2 * radius;

  float *in, *out, *weights;
  gpuErrCheck(gpuMallocManaged(&in, sizeof(float) * N));
  gpuErrCheck(gpuMallocManaged(&out, sizeof(float) * N));
  gpuErrCheck(gpuMallocManaged(&weights, sizeof(float) * numWeights));

  // Initialize input array.
  for (size_t i = 0; i < N; i++) {
    in[i] = 1.0f;
  }

  // Initialize weights for a simple averaging stencil.
  for (int i = 0; i < numWeights; i++) {
    weights[i] = 1.0f / numWeights;
  }

  int numBlocks = (N + blockSize - 1) / blockSize;

  auto Kernel = [
    =, radius = proteus::jit_variable(radius),
    tileSize = proteus::jit_variable(tileSize)
  ] __device__(float *weights) __attribute__((annotate("jit"))) {
    proteus::jit_array(weights, NWeights);
    float *tile = proteus::shared_array<float, MaxTileSize>(tileSize);
    int gid = getGlobalThreadIdX();
    int tid = getThreadIdX();

    tile[tid + radius] = in[gid];
    if (tid < radius) {
      tile[tid] = in[gid - radius];
      tile[tid + blockDim.x + radius] = in[gid + blockDim.x];
    }
    __syncthreads();

    float sum = 0.0f;
    for (int j = -radius; j <= radius; j++)
      sum += tile[tid + radius + j] * weights[radius + j];
    out[gid] = sum;
  };
  launchStencil(numBlocks, blockSize, kernel, weights);

  // Verify results. With all 1s input and averaging weights, output should be
  // ~1.0.
  bool passed = true;
  for (size_t i = radius; i < N - radius; i++) {
    if (out[i] < 0.99f || out[i] > 1.01f) {
      printf("FAIL: out[%zu] = %f, expected ~1.0\n", i, out[i]);
      passed = false;
      break;
    }
  }

  if (passed) {
    printf("PASSED\n");
  }

  gpuErrCheck(gpuFree(in));
  gpuErrCheck(gpuFree(out));
  gpuErrCheck(gpuFree(weights));

  proteus::finalize();
  return passed ? 0 : 1;
}

// clang-format off
// CHECK-FIRST: [LambdaSpec] Replacing slot {{[0-9]+}} with i32 2
// CHECK-FIRST: [LambdaSpec] Replacing slot {{[0-9]+}} with i32 68
// CHECK-FIRST: [SharedArray] Replace CB float* proteus::shared_array<float, 128ul, 0>(unsigned long, unsigned long) with @.proteus.shared
// CHECK: PASSED
// CHECK-FIRST: [proteus][JitEngineDevice] StorageCache rank 0 hits 0 accesses 1
// CHECK-SECOND: [proteus][JitEngineDevice] StorageCache rank 0 hits 1 accesses 1
// clang-format on
