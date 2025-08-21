#include <cstdlib>

#include <hip/hip_runtime.h>

#include "proteus/CppJitModule.hpp"

using namespace proteus;

int main() {
  const char *CPUCode = R"cpp(
    #include <cstdio>

    extern "C" void foo() {
      printf("Hello from JIT CPU code!\n");
    }
  )cpp";

  CppJitModule CJM("host", CPUCode);
  CJM.run<void>("foo");

  const char *GPUCode = R"cpp(
    #include <hip/hip_runtime.h>

    extern "C" __global__ void foo() {
      printf("Hello from JIT GPU code!\n");
    }
  )cpp";

  CppJitModule CJMGPU{"hip", GPUCode};
  CJMGPU.launch("foo", {1, 1, 1}, {1, 1, 1}, 0, nullptr);
  if (hipDeviceSynchronize() != HIP_SUCCESS)
    throw std::runtime_error("hipDeviceSynchronize failed");

  return 0;
}
