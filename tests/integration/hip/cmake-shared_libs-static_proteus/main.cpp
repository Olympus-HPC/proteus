#include <hip/hip_runtime.h>
#include <stdio.h>

void foo();
void bar();

__attribute__((annotate("jit"))) __global__ void kernel() {
  printf("kernel\n");
}

int main() {
  kernel<<<1, 1>>>();
  hipDeviceSynchronize();
  foo();
  bar();
  return 0;
}
