// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/cpp_source.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/cpp_source.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
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
// CHECK: [proteus][Dispatcher{{CUDA|HIP}}] MemoryCache procuid 0 hits 0 accesses 1
// CHECK: [proteus][Dispatcher{{CUDA|HIP}}] MemoryCache procuid 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: [proteus][Dispatcher{{CUDA|HIP}}] StorageCache procuid 0 hits 0 accesses 1
// CHECK-SECOND: [proteus][Dispatcher{{CUDA|HIP}}] StorageCache procuid 0 hits 1 accesses 1
