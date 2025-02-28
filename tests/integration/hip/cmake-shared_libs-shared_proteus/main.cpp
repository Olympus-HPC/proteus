#include <hip/hip_runtime.h>
#include <stdio.h>

#include <proteus/JitInterface.hpp>

void foo();
void bar();

__attribute__((annotate("jit"))) __global__ void kernel() {
  printf("kernel\n");
}

int main() {
  proteus::init();

  kernel<<<1, 1>>>();
  hipDeviceSynchronize();
  foo();
  bar();

  proteus::finalize();
  return 0;
}
