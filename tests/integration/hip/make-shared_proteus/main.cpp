#include <hip/hip_runtime.h>
#include <stdio.h>

#include <proteus/JitInterface.h>

__attribute__((annotate("jit"))) __global__ void kernel() {
  printf("kernel\n");
}

int main() {
  proteus::init();

  kernel<<<1, 1>>>();

  proteus::finalize();
  return 0;
}
