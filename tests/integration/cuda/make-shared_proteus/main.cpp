#include <proteus/JitInterface.h>

#include <cuda_runtime.h>

#include <cstdio>

__attribute__((annotate("jit"))) __global__ void kernel() {
  printf("kernel\n");
}

int main() {
  kernel<<<1, 1>>>();
  cudaDeviceSynchronize();

  return 0;
}
