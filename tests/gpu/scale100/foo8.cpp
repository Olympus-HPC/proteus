#include "../gpu_common.h"

__device__ void foo9Device0(int *, int *, int);

__device__ void foo8Device0(int *A, int *B, int N) {
  int Idx = blockIdx.x * blockDim.x + threadIdx.x;
  int Stride = gridDim.x * blockDim.x;

  for (int I = Idx; I < N; I += Stride)
    A[I] = A[I] + B[I];
}

__global__ __attribute__((annotate("jit"))) void foo8(int *A, int *B, int N) {
  int Idx = blockIdx.x * blockDim.x + threadIdx.x;
  int Stride = gridDim.x * blockDim.x;

  for (int I = Idx; I < N; I += Stride)
    A[I] = A[I] + B[I];

  foo9Device0(A, B, N);
}
