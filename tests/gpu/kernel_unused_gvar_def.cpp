#include "gpu_common.h"

__device__ int gvar = 23;

__global__ __attribute__((annotate("jit"))) void kernel_gvar() {
  gvar++;
  printf("Kernel gvar\n");
}
