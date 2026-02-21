#include <cstdlib>

#include <cuda_runtime.h>

#include "proteus/CppJitModule.h"
#include <proteus/JitInterface.h>

using namespace proteus;

int main() {
  const char *CPUCode = R"cpp(
    #include <cstdio>

    extern "C" void foo() {
      printf("Hello from JIT CPU code!\n");
    }
  )cpp";

  CppJitModule CJM{"host", CPUCode};
  auto Foo = CJM.getFunction<void()>("foo");
  Foo.run();

  const char *GPUCode = R"cpp(
    #include <cuda_runtime.h>

    extern "C" __global__ void foo() {
      printf("Hello from JIT GPU code!\n");
    }
  )cpp";

  CppJitModule CJMGPU{"cuda", GPUCode};
  auto FooKernel = CJMGPU.getKernel<void()>("foo");
  FooKernel.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr);
  if (cudaDeviceSynchronize() != cudaSuccess)
    throw std::runtime_error("cudaDeviceSynchronize failed");

  return 0;
}
