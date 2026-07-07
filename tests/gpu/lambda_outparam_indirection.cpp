// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="specialization;cache-stats" %build/lambda_outparam_indirection.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cstdio>

#include "gpu_common.h"
#include <proteus/JitInterface.h>

template <typename F>
__device__ __attribute__((noinline, optnone)) void forward_out_store(F **Dst,
                                                                     F *Src) {
  *Dst = Src;
}

template <typename F>
__device__ __attribute__((noinline, optnone)) void forward_out_memcpy(F **Dst,
                                                                      F *Src) {
  __builtin_memcpy(Dst, &Src, sizeof(Src));
}

template <typename F>
__global__ __attribute__((annotate("jit"))) void kernel_out_store(F Body,
                                                                  int *Out) {
  F *Chosen;
  forward_out_store(&Chosen, &Body);
  (*Chosen)();
  printf("store observed %d\n", *Out);
}

template <typename F>
__global__ __attribute__((annotate("jit"))) void kernel_out_memcpy(F Body,
                                                                   int *Out) {
  F *Chosen;
  forward_out_memcpy(&Chosen, &Body);
  (*Chosen)();
  printf("memcpy observed %d\n", *Out);
}

int main() {
  int *Out = nullptr;
  gpuErrCheck(gpuMallocManaged(&Out, sizeof(int)));
  *Out = -1;

  auto StoreBody = proteus::register_lambda(
      [Out, X = proteus::jit_variable(404)] __device__ {
        *Out = X;
        printf("store lambda %d\n", X);
      });
  kernel_out_store<<<1, 1>>>(StoreBody, Out);
  gpuErrCheck(gpuDeviceSynchronize());
  printf("host store %d\n", *Out);

  *Out = -1;
  auto MemcpyBody = proteus::register_lambda(
      [Out, X = proteus::jit_variable(505)] __device__ {
        *Out = X;
        printf("memcpy lambda %d\n", X);
      });
  kernel_out_memcpy<<<1, 1>>>(MemcpyBody, Out);
  gpuErrCheck(gpuDeviceSynchronize());
  printf("host memcpy %d\n", *Out);

  gpuErrCheck(gpuFree(Out));
  return 0;
}

// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 404
// CHECK: store lambda 404
// CHECK: store observed 404
// CHECK: host store 404
// CHECK-FIRST: [LambdaSpec] Replacing slot 0 with i32 505
// CHECK: memcpy lambda 505
// CHECK: memcpy observed 505
// CHECK: host memcpy 505
