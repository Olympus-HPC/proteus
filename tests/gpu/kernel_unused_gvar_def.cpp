#include "gpu_common.h"

__device__ int Gvar = 23;

__global__ __attribute__((annotate("jit"))) void kernelGvar() {
  Gvar++;
  printf("Kernel gvar\n");
}
