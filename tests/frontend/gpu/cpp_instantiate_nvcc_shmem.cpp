// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/cpp_instantiate_nvcc_shmem.%ext | %FILECHECK %s
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#if !PROTEUS_ENABLE_CUDA
#error "cpp_instantiate_nvcc_shmem requires PROTEUS_ENABLE_CUDA"
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

    template<int V>
    __global__ void shmem_kernel(int *Out) {
      extern __shared__ unsigned char Shmem[];
      if (threadIdx.x == 0) {
        Shmem[0] = static_cast<unsigned char>(V);
        Out[0] = static_cast<int>(Shmem[0]);
        printf("shmem V %d threads %u\n", V, static_cast<unsigned>(blockDim.x));
      }
    }

    extern "C" __global__ void shmem_plain(int *Out) {
      extern __shared__ unsigned char Shmem[];
      if (threadIdx.x == 0) {
        Shmem[0] = 11;
        Out[0] = static_cast<int>(Shmem[0]);
        printf("plain threads %u\n", static_cast<unsigned>(blockDim.x));
      }
    }
  )cpp";

  constexpr uint64_t LargeShmemSize = 49 * 1024;
  int *Out = nullptr;
  gpuErrCheck(gpuMallocManaged(&Out, sizeof(int)));
  *Out = 0;

  CppJitModule CJM{"cuda", Code, {}, CppJitCompilerBackend::Nvcc};
  auto &Inst = CJM.instantiate("shmem_kernel", "7");
  Inst.setFuncAttribute(CppJitFuncAttribute::MaxDynamicSharedMemorySize,
                        LargeShmemSize);

  gpuErrCheck(static_cast<gpuError_t>(
      Inst.launch({1, 1, 1}, {1, 1, 1}, LargeShmemSize, nullptr, Out)));
  gpuErrCheck(gpuDeviceSynchronize());

  std::cout << "Out " << *Out << "\n";

  *Out = 0;
  auto Kernel = CJM.getKernel<void(int *)>("shmem_plain");
  Kernel.setFuncAttribute(CppJitFuncAttribute::MaxDynamicSharedMemorySize,
                          LargeShmemSize);
  gpuErrCheck(static_cast<gpuError_t>(
      Kernel.launch({1, 1, 1}, {1, 1, 1}, LargeShmemSize, nullptr, Out)));
  gpuErrCheck(gpuDeviceSynchronize());

  std::cout << "Plain " << *Out << "\n";

  gpuErrCheck(gpuFree(Out));

  return 0;
}

// clang-format off
// CHECK: shmem V 7 threads 1
// CHECK: Out 7
// CHECK: plain threads 1
// CHECK: Plain 11
// clang-format on
