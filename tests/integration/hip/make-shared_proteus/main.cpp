#include <stdio.h>
#include <hip/hip_runtime.h>

__attribute__((annotate("jit")))
__global__ void kernel() {
    printf("kernel\n");
}

int main() {
    kernel<<<1,1>>>();
    return 0;
}
