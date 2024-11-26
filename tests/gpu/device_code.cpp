#include "gpu_common.h"
#include <cstdio>

__device__ void device_function(int a) { printf("Kernel %d\n", a); }
