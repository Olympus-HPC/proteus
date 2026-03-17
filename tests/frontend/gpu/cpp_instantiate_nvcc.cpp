// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="cache-stats" %build/cpp_instantiate_nvcc.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="cache-stats" %build/cpp_instantiate_nvcc.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#if !PROTEUS_ENABLE_CUDA
#error "cpp_instantiate_nvcc requires PROTEUS_ENABLE_CUDA"
#endif

#include "proteus/CppJitModule.h"

#include "../../gpu/gpu_common.h"

#include <iostream>

using namespace proteus;

int main() {
  const char *Code = R"cpp(
    #if !defined(__NVCC__)
    #error "Expected NVCC to compile instantiated CUDA module"
    #endif

    #include <cuda_runtime.h>
  #include <cstdio>
    #include <type_traits>

    template<int V>
    __global__ void foo(double A, double *R) {
      printf("foo V %d A %lf\n", V, A);
      *R = (A + 1);
    }

    template<typename T>
    __global__ void bar() {
      if constexpr (std::is_same_v<T, double>)
        printf("bar type double\n");
      else if constexpr (std::is_same_v<T, float>)
        printf("bar type float\n");
      else
        printf("bar type unhandled\n");
    }
  )cpp";

  double *R = nullptr;
  gpuErrCheck(gpuMallocManaged(&R, sizeof(double)));

  CppJitModule CJM{"cuda", Code, {}, CppJitCompilerBackend::Nvcc};
  auto &InstValue = CJM.instantiate("foo", "3");

  InstValue.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr, 42, R);
  gpuErrCheck(gpuDeviceSynchronize());
  std::cout << "Ret " << *R << "\n";

  InstValue.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr, 23, R);
  gpuErrCheck(gpuDeviceSynchronize());
  std::cout << "Ret " << *R << "\n";

  auto &InstTypeD = CJM.instantiate("bar", "double");
  InstTypeD.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr);
  gpuErrCheck(gpuDeviceSynchronize());

  auto &InstTypeF = CJM.instantiate("bar", "float");
  InstTypeF.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr);
  gpuErrCheck(gpuDeviceSynchronize());

  gpuErrCheck(gpuFree(R));

  return 0;
}

// clang-format off
// CHECK: foo V 3 A 42.000000
// CHECK: Ret 43
// CHECK: foo V 3 A 23.000000
// CHECK: Ret 24
// CHECK: bar type double
// CHECK: bar type float
// CHECK: [proteus][DispatcherHostCUDA] MemoryCache rank 0 hits 0 accesses 3
// CHECK-DAG: [proteus][DispatcherHostCUDA] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-DAG: [proteus][DispatcherHostCUDA] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-DAG: [proteus][DispatcherHostCUDA] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: [proteus][DispatcherHostCUDA] StorageCache rank 0 hits 3 accesses 6
// CHECK-SECOND: [proteus][DispatcherHostCUDA] StorageCache rank 0 hits 3 accesses 3
// clang-format on
