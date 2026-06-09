// clang-format off
// RUN: /opt/rocm-6.4.2/bin/amdclang++ -DPROTEUS_ENABLE_HIP -D__HIP_ROCclr__=1 -I/g/g11/bowen36/tmp/proteus/include -isystem /opt/rocm-6.4.2/include -O2 -DNDEBUG --offload-arch=gfx90a -fpass-plugin=%build/../../src/pass/libProteusPass.so -fpass-plugin=%build/../../src/pass/libLambdaPass.so -std=c++17 -x hip -c %s -o %t.o 2>&1 | %FILECHECK %s --check-prefix=CHECK-WARN
// clang-format on

#include <iostream>

#include "gpu_common.h"
#include "proteus/JitInterface.h"
template<typename T>
struct Functor {

__attribute__((noinline))
__host__ __device__
void operator()(T&& Lam, const int I) {
  Lam(I);
}
};

template <typename T>
__global__ __attribute__((annotate("jit"))) void kernel1d(int N, T LB) {
  // Functor<std::decay_t<T>> f;
  // Functor<T> f;
  const int I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I < N) {
    LB(I);
  }
}

template <typename T> void launch(int N, T &&LB) {
  auto RegisteredFunctor = proteus::register_lambda(LB);
  kernel1d<<<1, 32>>>(N, RegisteredFunctor);
  // kernel1d<<<1, 32>>>(N, LB);
  gpuErrCheck(gpuDeviceSynchronize());
}

int main() {
  int *X = nullptr;
  gpuErrCheck(gpuMallocManaged(&X, sizeof(int)));
  X[0] = -1;

  launch(32, [=] __device__ __attribute__((annotate("jit"))) (int I) {
    // The warning is triggered by LDS/shared-memory globals defined in the
    // registered lambda body being used through the non-kernel Proteus wrapper.
     __shared__ int Smem[32];
    Smem[threadIdx.x] = I;
    __syncthreads();
    if (threadIdx.x == 0) {
      X[0] = Smem[0];
    }
  });

  std::cout << "x[0] = " << X[0] << "\n";
  gpuErrCheck(gpuFree(X));
  return 0;
}

// CHECK-WARN: warning: local memory global used by non-kernel function
