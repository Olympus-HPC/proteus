#include <hip/hip_runtime.h>

__global__ void kernelBar() { printf("libbar kernel\n"); }

void bar() {
  kernelBar<<<1, 1>>>();
  hipDeviceSynchronize();
}
