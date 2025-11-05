#include <stdio.h>

#include "gpu_common.h"

__device__ int value;

__global__ __attribute__((annotate("jit"))) static void kernel2() {
  printf("Kernel 2 value is %d\n", value);
}

void print_gval2(int hValue) {
  void *dAddr;
  gpuErrCheck(gpuGetSymbolAddress(&dAddr, value));
  gpuErrCheck(gpuMemcpy(dAddr, &hValue, sizeof(int), gpuMemcpyHostToDevice));
  kernel2<<<1, 1>>>();
  gpuErrCheck(gpuDeviceSynchronize());
}
