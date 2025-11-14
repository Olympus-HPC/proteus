// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/cpp_host_device | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/cpp_host_device | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include "proteus/CppJitModule.hpp"

#if PROTEUS_ENABLE_HIP
#define TARGET "host_hip"
#define PREFIX "hip"
#define INCLUDE "#include <hip/hip_runtime.h>"
#define DEF_CHECK                                                              \
  R"cpp(#define CHECK(c) {                                                     \
    hipError_t e = c;                                                          \
    if (e != hipSuccess)                                                       \
      printf("Error on %s:%d => %s\n", __FILE__, __LINE__,                     \
             hipGetErrorString(e));                                            \
})cpp"

#elif PROTEUS_ENABLE_CUDA
#define TARGET "host_cuda"
#define PREFIX "cuda"
#define INCLUDE "#include <cuda_runtime.h>"
#define DEF_CHECK                                                              \
  R"cpp(#define CHECK(c) {                                                     \
    cudaError_t e = c;                                                         \
    if (e != cudaSuccess)                                                      \
      printf("Error on %s:%d => %s\n", __FILE__, __LINE__,                     \
             cudaGetErrorString(e));                                           \
})cpp"

#else
#error "Expected PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA defined"
#endif

using namespace proteus;

int main() {
  const char *Code = INCLUDE "\n" DEF_CHECK "\n"
                             R"cpp(
    #include <cstdio>

    __global__ void kernel() {
      printf("Hello from kernel\n");
    }

    extern "C" void foo() {
      kernel<<<1,1>>>();
      CHECK()cpp" PREFIX R"cpp(DeviceSynchronize());

      printf("hello from host\n");
    }

    template<typename Lambda>
    __global__ void forall_kernel(size_t N, Lambda L) {
      int tid = threadIdx.x + blockIdx.x * blockDim.x;
      int stride = blockDim.x * gridDim.x;
      for (size_t i = tid; i < N; i += stride)
        L(i);
    }

    template<typename Lambda>
    void forall(size_t N, Lambda &&L) {
      int blocks = 256;
      int grids = (N + blocks - 1)/blocks;
      forall_kernel<<<grids, blocks>>>(N, L);
    }

    extern "C" void bar() {
      double *a;
      CHECK()cpp" PREFIX R"cpp(MallocManaged(&a, sizeof(double)*1024));
      for(int i=0; i<1024; ++i)
        a[i] = i;

      forall(1024, [=] __device__ (int i) {
        a[i] = a[i]*a[i];
      });

      CHECK()cpp" PREFIX R"cpp(DeviceSynchronize());

      printf("a[0] = %lf\n", a[0]);
      printf("a[4] = %lf\n", a[4]);
      printf("a[16] = %lf\n", a[16]);
      printf("a[1023] = %lf\n", a[1023]);

      CHECK()cpp" PREFIX R"cpp(Free(a));
    }
   )cpp";

  CppJitModule CJM{TARGET, Code};
  CJM.compile();
  auto Foo = CJM.getFunction<void()>("foo");
  Foo.run();
  auto Bar = CJM.getFunction<void()>("bar");
  Bar.run();

  return 0;
}

// clang-format off
// CHECK: Hello from kernel
// CHECK: hello from host
// CHECK: a[0] = 0.000000
// CHECK: a[4] = 16.000000
// CHECK: a[16] = 256.000000
// CHECK: a[1023] = 1046529.000000
// CHECK-COUNT-2: [proteus][DispatcherHost] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: [proteus][DispatcherHost] StorageCache rank 0 hits 1 accesses 2
// CHECK-SECOND: [proteus][DispatcherHost] StorageCache rank 0 hits 1 accesses 1
