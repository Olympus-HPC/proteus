// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="cache-stats" %build/cpp_device_nvcc.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" PROTEUS_TRACE_OUTPUT="cache-stats" %build/cpp_device_nvcc.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#if !PROTEUS_ENABLE_CUDA
#error "cpp_device_nvcc requires PROTEUS_ENABLE_CUDA"
#endif

#include "proteus/CppJitModule.h"

#include "../../gpu/gpu_common.h"

#include <cstdio>
#include <cstdlib>

using namespace proteus;

int main() {
  const char *Code = R"cpp(
    #include <cuda_runtime.h>

    extern "C" __global__ void write_int(int *out) {
      if (blockIdx.x == 0 && threadIdx.x == 0)
        out[0] = 123;
    }
  )cpp";

  int *DevOut = nullptr;
  gpuErrCheck(gpuMalloc(&DevOut, sizeof(int)));
  int Zero = 0;
  gpuErrCheck(gpuMemcpy(DevOut, &Zero, sizeof(int), gpuMemcpyHostToDevice));

  CppJitModule CJM{"cuda", Code, {}, CppJitCompilerBackend::Nvcc};
  auto Kernel = CJM.getKernel<void(int *)>("write_int");
  Kernel.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr, DevOut);
  gpuErrCheck(gpuDeviceSynchronize());

  int HostOut = 0;
  gpuErrCheck(gpuMemcpy(&HostOut, DevOut, sizeof(int), gpuMemcpyDeviceToHost));
  gpuErrCheck(gpuFree(DevOut));

  std::printf("HostOut %d\n", HostOut);
  if (HostOut != 123)
    std::abort();

  return 0;
}

// clang-format off
// CHECK: HostOut 123
// CHECK: [proteus][DispatcherCUDA] MemoryCache rank 0 hits 0 accesses 1
// CHECK: [proteus][DispatcherCUDA] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK: [proteus][DispatcherCUDA] ObjectCacheChain rank 0 with 1 level(s):
// CHECK-FIRST: [proteus][DispatcherCUDA] StorageCache rank 0 hits 1 accesses 2
// CHECK-SECOND: [proteus][DispatcherCUDA] StorageCache rank 0 hits 1 accesses 1
// clang-format on
