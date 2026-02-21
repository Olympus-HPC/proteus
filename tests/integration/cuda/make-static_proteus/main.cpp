#include <cuda_runtime.h>
#include <stdio.h>

#include <proteus/JitInterface.h>

__attribute__((annotate("jit"))) __global__ void kernel() {
  printf("kernel\n");
}

int main() {
  kernel<<<1, 1>>>();

  return 0;
}
