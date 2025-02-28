#include "gpu_common.h"
#include <cstdio>

__device__ void deviceFunction(int A) { printf("device_func %d\n", A); }
