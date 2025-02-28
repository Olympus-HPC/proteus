#include <hip/hip_runtime.h>

__global__ void kernelFoo() { printf("libfoo kernel\n"); }

void foo() {
  kernelFoo<<<1, 1>>>();
  hipDeviceSynchronize();
}
