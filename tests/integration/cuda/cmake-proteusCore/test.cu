#include <cuda_runtime.h>

extern "C" __global__ void foo(int a) {
  printf("Kernel, arg %d\n", a);
}
