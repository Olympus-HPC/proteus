#include <stdio.h>

#include "gpu_common.h"
#include <proteus/JitInterface.hpp>

__global__ __attribute__((annotate("jit"))) static void kernel() {
  printf("File2 Kernel\n");
}

void foo() {
  kernel<<<1, 1>>>();
  gpuErrCheck(gpuDeviceSynchronize());
}
