// NOLINTBEGIN

// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/floydwarshall.%ext 1024 100 16 | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/floydwarshall.%ext 1024 100 16 | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <proteus/Frontend/Builtins.hpp>
#include <proteus/JitFrontend.hpp>

#include "../../gpu/gpu_common.h"

#include <assert.h>
#include <chrono>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

constexpr unsigned int MAXDISTANCE = 200;

/*
 * The floyd Warshall algorithm is a multipass algorithm
 * that calculates the shortest path between each pair of
 * nodes represented by pathDistanceBuffer.
 *
 * In each pass a node k is introduced and the pathDistanceBuffer
 * which has the shortest distance between each pair of nodes
 * considering the (k-1) nodes (that are introduced in the previous
 * passes) is updated such that
 *
 * ShortestPath(x,y,k) = min(ShortestPath(x,y,k-1), ShortestPath(x,k,k-1) +
 * ShortestPath(k,y,k-1)) where x and y are the pair of nodes between which the
 * shortest distance is being calculated.
 *
 * pathBuffer stores the intermediate nodes through which the shortest
 * path goes for each pair of nodes.
 *
 * numNodes is the number of nodes in the graph.
 *
 * for more detailed explaination of the algorithm kindly refer to the document
 * provided with the sample
 */

using namespace proteus;
using namespace builtins::gpu;

#if PROTEUS_ENABLE_HIP
#define TARGET "hip"
#elif PROTEUS_ENABLE_CUDA
#define TARGET "cuda"
#else
#error "Expected PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA defined"
#endif

__global__ void floydWarshallPass(unsigned int *__restrict__ pathDistanceBuffer,
                                  unsigned int *__restrict__ pathBuffer,
                                  const unsigned int numNodes,
                                  const unsigned int pass) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int n2 = numNodes * numNodes;
  if (idx >= n2)
    return;

  unsigned int yValue = idx / numNodes;
  unsigned int xValue = idx - yValue * numNodes;
  unsigned int k = pass;

  unsigned int oldWeight = pathDistanceBuffer[yValue * numNodes + xValue];
  unsigned int tempWeight = pathDistanceBuffer[yValue * numNodes + k] +
                            pathDistanceBuffer[k * numNodes + xValue];

  if (tempWeight < oldWeight) {
    pathDistanceBuffer[yValue * numNodes + xValue] = tempWeight;
    pathBuffer[yValue * numNodes + xValue] = k;
  }
}

auto createJitModuleSpecial(unsigned int _numNodes) {
  auto J = std::make_unique<JitModule>(TARGET);
  auto KernelHandle =
      J->addKernelTT<void(unsigned int *, unsigned int *, unsigned int,
                          unsigned int)>("floydWarshallPass");
  auto &F = KernelHandle.F;
  auto [pathDistanceBuffer, pathBuffer, numNodes, pass] = F.getArgs();

  F.beginFunction();
  {
    // Bake only numNodes as a runtime constant; pass is a dynamic kernel arg
    auto rcNumNodes = F.defRuntimeConst<unsigned int>(_numNodes, "numNodes");

    auto idx = F.declVar<unsigned int>("idx");
    auto totThreads = F.declVar<unsigned int>("totThreads");
    auto xValue = F.declVar<unsigned int>("xValue");
    auto yValue = F.declVar<unsigned int>("yValue");
    auto k = F.declVar<unsigned int>("k");
    auto oldWeight = F.declVar<unsigned int>("oldWeight");
    auto tempWeight = F.declVar<unsigned int>("tempWeight");

    auto tidx = F.callBuiltin(getThreadIdX);
    auto bidx = F.callBuiltin(getBlockIdX);
    auto bdim = F.callBuiltin(getBlockDimX);
    auto gdim = F.callBuiltin(getGridDimX);

    idx = bidx * bdim + tidx;
    totThreads = gdim * bdim;

    // Flattened 1D thread id maps to 2D (y, x)
    yValue = idx / rcNumNodes;
    xValue = idx - yValue * rcNumNodes;
    k = pass;

    // Guard against overrun when grid > N*N
    auto n2 = F.declVar<unsigned int>("n2");
    n2 = rcNumNodes * rcNumNodes;
    F.beginIf(idx < n2);
    {
      oldWeight = pathDistanceBuffer[yValue * rcNumNodes + xValue];
      tempWeight = pathDistanceBuffer[yValue * rcNumNodes + k] +
                   pathDistanceBuffer[k * rcNumNodes + xValue];

      F.beginIf(tempWeight < oldWeight);
      {
        pathDistanceBuffer[yValue * rcNumNodes + xValue] = tempWeight;
        pathBuffer[yValue * rcNumNodes + xValue] = k;
      }
      F.endIf();
    }
    F.endIf();
    F.ret();
  }
  F.endFunction();

  return std::make_pair(std::move(J), KernelHandle);
}

int main(int argc, char **argv) {
  if (argc != 4) {
    printf("Usage: %s <number of nodes> <iterations> <block size>\n", argv[0]);
    return 1;
  }

  int numNodes = atoi(argv[1]);
  int numIterations = atoi(argv[2]);
  int blockSize = atoi(argv[3]);

  if (numNodes % blockSize != 0) {
    numNodes = (numNodes / blockSize + 1) * blockSize;
  }

  const size_t matrixSizeBytes = static_cast<size_t>(numNodes) *
                                 static_cast<size_t>(numNodes) *
                                 sizeof(unsigned int);

  unsigned int *pathDistanceMatrix = (unsigned int *)malloc(matrixSizeBytes);
  assert(pathDistanceMatrix != nullptr);
  unsigned int *pathMatrix = (unsigned int *)malloc(matrixSizeBytes);
  assert(pathMatrix != nullptr);

  srand(123);
  for (int i = 0; i < numNodes; i++) {
    for (int j = 0; j < numNodes; j++) {
      int index = i * numNodes + j;
      pathDistanceMatrix[index] =
          static_cast<unsigned int>(1 + (rand() % MAXDISTANCE));
    }
  }
  for (int i = 0; i < numNodes; ++i) {
    int iXWidth = i * numNodes;
    pathDistanceMatrix[iXWidth + i] = 0u;
  }

  for (int i = 0; i < numNodes; ++i) {
    for (int j = 0; j < i; ++j) {
      pathMatrix[i * numNodes + j] = static_cast<unsigned int>(i);
      pathMatrix[j * numNodes + i] = static_cast<unsigned int>(j);
    }
    pathMatrix[i * numNodes + i] = static_cast<unsigned int>(i);
  }

  // Print initial matrix values for verification
  for (int i = 0; i < 10 && i < numNodes * numNodes; i++) {
    printf("init pathDistanceMatrix[%d] = %u\n", i, pathDistanceMatrix[i]);
  }

  if (blockSize * blockSize > 256) {
    blockSize = 16;
  }

  const int threadsPerBlock = blockSize * blockSize; // 1D launch
  const uint64_t totalElems = (uint64_t)numNodes * (uint64_t)numNodes;
  const int numBlocks =
      static_cast<int>((totalElems + threadsPerBlock - 1) / threadsPerBlock);

  unsigned int *pathDistanceBuffer = nullptr;
  unsigned int *pathBuffer = nullptr;
  gpuErrCheck(gpuMalloc((void **)&pathDistanceBuffer, matrixSizeBytes));
  gpuErrCheck(gpuMalloc((void **)&pathBuffer, matrixSizeBytes));

  float total_time_s = 0.f;

  printf("Creating JIT module\n");
  auto [J, KernelHandle] =
      createJitModuleSpecial(static_cast<unsigned int>(numNodes));
  printf("Compiling JIT module\n");
  J->compile();

  for (int n = 0; n < numIterations; n++) {
    gpuErrCheck(gpuMemcpy(pathDistanceBuffer, pathDistanceMatrix,
                          matrixSizeBytes, gpuMemcpyHostToDevice));

    gpuErrCheck(gpuDeviceSynchronize());
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < numNodes; i++) {
      // floydWarshallPass<<<numBlocks, threadsPerBlock>>>(pathDistanceBuffer,
      // pathBuffer, numNodes, i);

      gpuErrCheck(KernelHandle.launch(
          {static_cast<unsigned>(numBlocks), 1U, 1U},
          {static_cast<unsigned>(threadsPerBlock), 1U, 1U}, 0, nullptr,
          pathDistanceBuffer, pathBuffer, static_cast<unsigned>(numNodes),
          static_cast<unsigned>(i)));
    }

    gpuErrCheck(gpuDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();
    auto time_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    total_time_s += static_cast<float>(time_ns * 1e-9);
  }

  printf("Average kernel execution time %f (ms)\n",
         total_time_s / numIterations * 1e3f);

  gpuErrCheck(gpuMemcpy(pathDistanceMatrix, pathDistanceBuffer, matrixSizeBytes,
                        gpuMemcpyDeviceToHost));

  // Print sections for FileCheck: first 5, middle 5, last 5
  {
    size_t te = static_cast<size_t>(totalElems);

    // First 5
    size_t firstCount = te < 5 ? te : 5;
    for (size_t i = 0; i < firstCount; ++i) {
      printf("dist[%zu] = %u\n", i, pathDistanceMatrix[i]);
    }

    // Middle 5 (centered around te/2)
    size_t midStart = 0;
    if (te > 0) {
      size_t half = te / 2;
      midStart = (half > 2) ? (half - 2) : 0;
      if (midStart + 5 > te) {
        midStart = te > 5 ? (te - 5) : 0;
      }
    }
    size_t midCount = (te - midStart) < 5 ? (te - midStart) : 5;
    for (size_t i = 0; i < midCount; ++i) {
      size_t idx = midStart + i;
      printf("dist[%zu] = %u\n", idx, pathDistanceMatrix[idx]);
    }

    // Last 5
    size_t lastStart = te > 5 ? (te - 5) : 0;
    size_t lastCount = te < 5 ? te : 5;
    for (size_t i = 0; i < lastCount; ++i) {
      size_t idx = lastStart + i;
      printf("dist[%zu] = %u\n", idx, pathDistanceMatrix[idx]);
    }
  }

  gpuErrCheck(gpuFree(pathDistanceBuffer));
  gpuErrCheck(gpuFree(pathBuffer));

  free(pathDistanceMatrix);
  free(pathMatrix);
  return 0;
}

//clang-format off
// CHECK: init pathDistanceMatrix[0] = 0
// CHECK-NEXT: init pathDistanceMatrix[1] = 14
// CHECK-NEXT: init pathDistanceMatrix[2] = 74
// CHECK-NEXT: init pathDistanceMatrix[3] = 31
// CHECK-NEXT: init pathDistanceMatrix[4] = 80
// CHECK-NEXT: init pathDistanceMatrix[5] = 132
// CHECK-NEXT: init pathDistanceMatrix[6] = 196
// CHECK-NEXT: init pathDistanceMatrix[7] = 23
// CHECK-NEXT: init pathDistanceMatrix[8] = 27
// CHECK-NEXT: init pathDistanceMatrix[9] = 2
// CHECK-NEXT: Creating JIT module
// CHECK-NEXT: Compiling JIT module
// CHECK-NEXT: Average kernel execution time {{.*}} (ms)
// CHECK-NEXT: dist[0] = 0
// CHECK-NEXT: dist[1] = 4
// CHECK-NEXT: dist[2] = 5
// CHECK-NEXT: dist[3] = 5
// CHECK-NEXT: dist[4] = 5
// CHECK-NEXT: dist[524286] = 4
// CHECK-NEXT: dist[524287] = 2
// CHECK-NEXT: dist[524288] = 4
// CHECK-NEXT: dist[524289] = 3
// CHECK-NEXT: dist[524290] = 2
// CHECK-NEXT: dist[1048571] = 5
// CHECK-NEXT: dist[1048572] = 4
// CHECK-NEXT: dist[1048573] = 4
// CHECK-NEXT: dist[1048574] = 4
// CHECK-NEXT: dist[1048575] = 0
// The KernelHandle stores the kernel function avoiding cache lookup
// CHECK: JitCache hits 0 total 1
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
// CHECK: JitCache hits 0 total 0
// CHECK: JitStorageCache hits 0 total 0

// NOLINTEND
