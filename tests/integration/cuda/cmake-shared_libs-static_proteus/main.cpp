#include <cuda_runtime.h>

#include <cstdio>

#include <proteus/JitInterface.h>

void foo();
void bar();

__attribute__((annotate("jit"))) __global__ void kernel() {
  printf("kernel\n");
}

int main() {
  kernel<<<1, 1>>>();
  cudaDeviceSynchronize();
  foo();
  bar();

  return 0;
}
