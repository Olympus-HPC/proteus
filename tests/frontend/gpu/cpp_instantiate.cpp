// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/cpp_instantiate.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/cpp_instantiate.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
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
    template<int V>
    __global__ void foo(double A, double *R) {
        printf("foo V %d A %lf\n", V, A);
        *R = (A+1);
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

  double *R;
  gpuErrCheck(gpuMallocManaged(&R, sizeof(double)));

  CppJitModule CJM{TARGET, Code};
  auto InstValue = CJM.instantiate("foo", "3");

  InstValue.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr, 42, R);
  gpuErrCheck(gpuDeviceSynchronize());
  std::cout << "Ret " << *R << "\n";

  InstValue.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr, 23, R);
  gpuErrCheck(gpuDeviceSynchronize());
  std::cout << "Ret " << *R << "\n";

  auto InstTypeD = CJM.instantiate("bar", "double");
  InstTypeD.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr);
  gpuErrCheck(gpuDeviceSynchronize());

  auto InstTypeF = CJM.instantiate("bar", "float");
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
// CHECK: JitCache hits 0 total 3
// CHECK-DAG: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-DAG: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-DAG: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 3
// CHECK-SECOND: JitStorageCache hits 3 total 3
