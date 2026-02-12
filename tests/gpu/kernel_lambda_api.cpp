// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_TRACE_OUTPUT="specialization" PROTEUS_CACHE_DIR="%t.$$.proteus" %build/kernel_lambda_api.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/kernel_lambda_api.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <climits>
#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

template <typename BODY, typename IndexType, typename Params>
__global__ void kernel(const BODY Body, const IndexType Length, Params Param) {
  proteus::jit_arg(Length);

  IndexType I = blockDim.x * blockIdx.x + threadIdx.x;
  if (I < Length)
    Body(I);

  if (Param) {
    printf("Got param\n");
  }
}

template <typename BODY, typename IndexType, typename Params>
void run(BODY &&Body, IndexType Length, Params Param) {
  constexpr int BlockSize = 256;
  const int NumBlocks = (Length + BlockSize - 1) / BlockSize;
  kernel<<<NumBlocks, BlockSize>>>(Body, Length, Param);
  gpuErrCheck(gpuDeviceSynchronize());
}

int main() {
  int *Param = nullptr;
  run(
      [=] __device__(size_t I) {
        if ((I % 256) == 0)
          printf("Kernel Index %lu/%d\n", I, 1024);
      },
      1024UL, Param);

  return 0;
}

// clang-format off
// CHECK-FIRST: [ObjectCacheChain] Added cache level: storage
// CHECK-FIRST-NEXT: [ObjectCacheChain] Chain for JitEngineDevice: Storage
// CHECK-FIRST: [ArgSpec] Replaced Function _Z6kernelIZ4mainEUlmE_mPiEvT_T0_T1_{{[^ ]*}} ArgNo 1 with value i64 1024
// CHECK-DAG: Kernel Index 256/1024
// CHECK-DAG: Kernel Index 512/1024
// CHECK-DAG: Kernel Index 768/1024
// CHECK-DAG: Kernel Index 0/1024
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 hits 0 accesses 1
// CHECK: [proteus][JitEngineDevice] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: [proteus][JitEngineDevice] ObjectCacheChain rank 0 with 1 level(s):
// CHECK-FIRST: [proteus][JitEngineDevice] StorageCache rank 0 hits 0 accesses 1
// CHECK-SECOND: [proteus][JitEngineDevice] StorageCache rank 0 hits 1 accesses 1
