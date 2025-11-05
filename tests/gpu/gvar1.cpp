#include <stdio.h>

#include "gpu_common.h"

__device__ int value;

__global__ __attribute__((annotate("jit"))) static void kernel1() {
  printf("Kernel 1 value is %d\n", value);
}

void print_gval1(int hValue) {
  void *dAddr;
  gpuErrCheck(gpuGetSymbolAddress(&dAddr, value));
  gpuErrCheck(gpuMemcpy(dAddr, &hValue, sizeof(int), gpuMemcpyHostToDevice));
  kernel1<<<1, 1>>>();
  gpuErrCheck(gpuDeviceSynchronize());
}
