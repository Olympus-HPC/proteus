#include <hip/hip_runtime.h>

__global__ void kernel_foo() { printf("libfoo kernel\n"); }

void foo() {
  kernel_foo<<<1, 1>>>();
  hipDeviceSynchronize();
}
