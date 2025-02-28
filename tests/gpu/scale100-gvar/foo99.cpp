#include "../gpu_common.h"

extern __device__ int Gvar0;

__device__ void foo0Device0(int *, int *, int);

__device__ void foo99Device0(int *A, int *B, int N) {
  int Idx = blockIdx.x * blockDim.x + threadIdx.x;
  int Stride = gridDim.x * blockDim.x;

  for (int I = Idx; I < N; I += Stride)
    A[I] = A[I] + B[I];
}

__global__ __attribute__((annotate("jit"))) void foo99(int *A, int *B, int N) {
  int Idx = blockIdx.x * blockDim.x + threadIdx.x;
  int Stride = gridDim.x * blockDim.x;

  for (int I = Idx; I < N; I += Stride)
    A[I] = A[I] + B[I];

  foo0Device0(A, B, N);

  if (Idx == 0) {

    atomicAdd(&Gvar0, 1);
  }
}
