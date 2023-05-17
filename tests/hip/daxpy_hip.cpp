#include <cstdlib>
#include <cstddef>
#include <iostream>

#include <hip/hip_runtime.h>

__global__
__attribute__((annotate("jit", 4)))
void 
daxpy_impl(double a, double* x, double* y, int N)
{
    std::size_t i = blockIdx.x * 256 + threadIdx.x;
    if (i < N) {
         y[i] += x[i]*a; 
    }
}

void daxpy(double a, double* x, double* y, int N)
{
      const std::size_t grid_size = (((N) + (256)-1) / (256));
      hipLaunchKernelGGL((daxpy_impl),dim3(grid_size), dim3(256), 0, 0, a, x, y, N );
}

int main(int argc, char** argv) {
    int N = 1024;
    double* x;
    double* y;

    hipMallocManaged(&x, sizeof(double)*N);
    hipMallocManaged(&y, sizeof(double)*N);

    for (std::size_t i{0}; i < N; i++) {
        x[i] = 0.31414*i;
        y[i] = 0.0;
    }

    std::cout << y[10] << std::endl;
    daxpy(6.2, x, y, N);
    hipDeviceSynchronize();
    std::cout << y[10] << std::endl;
    daxpy(6.2, x, y, N);
    hipDeviceSynchronize();
    std::cout << y[10] << std::endl;


    hipFree(x);
    hipFree(y);
}