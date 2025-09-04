// clang-format off
// RUN: rm -rf .proteus
// RUN: ./cpp_same_symbol.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./cpp_same_symbol.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf .proteus
// clang-format on

// Tests that kernels with the same symbol from different modules are correctly
// cached and execute individually.

#include "proteus/CppJitModule.hpp"

#include "../../gpu/gpu_common.h"

#if PROTEUS_ENABLE_HIP
#define TARGET "hip"
#define INCLUDE "#include <hip/hip_runtime.h>"

#elif PROTEUS_ENABLE_CUDA
#define TARGET "cuda"
#define INCLUDE "#include <cuda_runtime.h>"

#else
#error "Expected PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA defined"
#endif

using namespace proteus;

int main() {
  const char *Code = INCLUDE R"cpp(
    __global__
    extern "C" void foo(int a) {
        printf("Kernel %d\n", a);
    }
   )cpp";

  CppJitModule CJM{TARGET, Code};
  auto Kernel = CJM.getKernel<void(int)>("foo");
  Kernel.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr, 42);
  gpuErrCheck(gpuDeviceSynchronize());

  const char *Code2 = INCLUDE R"cpp(
    __global__
    extern "C" void foo(int a) {
        printf("Other Kernel %d\n", a);
    }
   )cpp";

  CppJitModule CJM2(TARGET, Code2);
  auto Kernel2 = CJM2.getKernel<void(int)>("foo");
  Kernel2.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr, 42);
  gpuErrCheck(gpuDeviceSynchronize());
}

// clang-format off
// CHECK: Kernel 42
// CHECK: Other Kernel 42
// CHECK: JitCache hits 0 total 2
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 2
// CHECK-SECOND: JitStorageCache hits 2 total 2
