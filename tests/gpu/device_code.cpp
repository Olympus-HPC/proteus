#include "gpu_common.h"
#include <cstdio>

__device__ void device_function(int a) { printf("device_func %d\n", a); }

__global__ void kernel2() { printf("Kernel2\n"); }
