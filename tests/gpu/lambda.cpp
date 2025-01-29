#include <iostream>

#include "JitVariable.hpp"

#include "gpu_common.h"

template<typename T>
__global__ __attribute__((annotate("jit"))) void kernel(T LB) {
//__global__ void kernel(T LB) {
  std::size_t i = blockIdx.x + threadIdx.x;
  if (i == 0)
    LB();
}

template <typename T> 
void run(T&& LB) {
  proteus::register_lambda(LB);
  kernel<<<1,1>>>(LB);
  gpuErrCheck(gpuDeviceSynchronize());
}

int main(int argc, char **argv) {
  double a{3.14};
  double *x;
  gpuErrCheck(gpuMallocManaged(&x, sizeof(double)*2));

  run([=, a = proteus::jit_variable(a)] __device__ __attribute__((annotate("jit"))) () { x[0] = a; });

  gpuErrCheck(gpuFree(x));
}
