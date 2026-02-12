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

__attribute__((annotate("jit", 3, 4))) __global__ void
stencil1d(float *out, float *in, size_t N, int radius, float *weights) {
  proteus::jit_array(weights, NWeights);
  float *tile = proteus::shared_array<float, 10>(68);
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
}

int main() {
  proteus::init();

  const size_t N = 256;
  const int radius = 2;
  const int blockSize = 64;
  const int numWeights = 2 * radius + 1;

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

  // Calculate shared memory size: blockSize + 2*radius elements.
  size_t sharedMemSize = (blockSize + 2 * radius) * sizeof(float);
  int numBlocks = (N + blockSize - 1) / blockSize;

#if PROTEUS_ENABLE_HIP
  hipLaunchKernelGGL(stencil1d, dim3(numBlocks), dim3(blockSize), sharedMemSize,
                     0, out, in, N, radius, weights);
#elif PROTEUS_ENABLE_CUDA
  stencil1d<<<numBlocks, blockSize, sharedMemSize>>>(out, in, N, radius,
                                                     weights);
#endif
  gpuErrCheck(gpuDeviceSynchronize());

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
// CHECK-FIRST: [ArgSpec] Replaced Function {{.*}}stencil1d{{.*}} ArgNo 2 with value i64 256
// CHECK-FIRST: [ArgSpec] Replaced Function {{.*}}stencil1d{{.*}} ArgNo 3 with value i32 2
// CHECK: PASSED
// CHECK-FIRST: [proteus][JitEngineDevice] StorageCache rank 0 hits 0 accesses 1
// CHECK-SECOND: [proteus][JitEngineDevice] StorageCache rank 0 hits 1 accesses 1
// clang-format on
