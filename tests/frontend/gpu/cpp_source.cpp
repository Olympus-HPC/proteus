// clang-format off
// RUN: rm -rf .proteus
// RUN: ./cpp_source.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./cpp_source.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf .proteus
// clang-format on

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
}

// clang-format off
// CHECK: Kernel 42
// CHECK: JitCache hits 0 total 1
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
