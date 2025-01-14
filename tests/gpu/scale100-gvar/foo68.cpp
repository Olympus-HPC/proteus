#include "../gpu_common.h"


extern __device__ int gvar0;


__device__ void foo69_device0(int *, int *, int);

__device__ void foo68_device0(int *a, int *b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for(int i=idx; i<n; i+=stride)
        a[i] = a[i] + b[i];
}

__global__ __attribute__((annotate("jit"))) void foo68 (int *a, int *b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for(int i=idx; i<n; i+=stride)
        a[i] = a[i] + b[i];

    foo69_device0(a, b, n);

    
    if(idx == 0) {
        
        atomicAdd(&gvar0, 1);
    }
    
}

