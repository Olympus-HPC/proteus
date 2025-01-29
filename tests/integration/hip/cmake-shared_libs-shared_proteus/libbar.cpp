#include <hip/hip_runtime.h>

__global__ void kernel_bar() {
    printf("libbar kernel\n");
}

void bar() {
    kernel_bar<<<1,1>>>();
    hipDeviceSynchronize();
}
