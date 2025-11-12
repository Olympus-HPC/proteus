#include <stdio.h>

#include "gpu_common.h"

__device__ int Value;

__global__ __attribute__((annotate("jit"))) static void kernel2() {
  printf("Kernel 2 value is %d\n", Value);
}

void printGVal2(int HValue) {
  void *DAddr;
  gpuErrCheck(gpuGetSymbolAddress(&DAddr, Value));
  gpuErrCheck(gpuMemcpy(DAddr, &HValue, sizeof(int), gpuMemcpyHostToDevice));
  kernel2<<<1, 1>>>();
  gpuErrCheck(gpuDeviceSynchronize());
}
