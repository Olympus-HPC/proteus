// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/mlir_gpu_jit_module_source.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/mlir_gpu_jit_module_source.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cassert>
#include <cstdio>

#include <proteus/MLIRJitModule.h>

#include "../../../gpu/gpu_common.h"

using namespace proteus;

int main() {
#if defined(PROTEUS_ENABLE_CUDA)
  const char *Target = "cuda";
#elif defined(PROTEUS_ENABLE_HIP)
  const char *Target = "hip";
#else
  return 0;
#endif

  static constexpr const char *Code = R"mlir(
module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @write42(%out: !llvm.ptr) kernel {
      %c42 = arith.constant 42 : i32
      llvm.store %c42, %out : i32, !llvm.ptr
      gpu.return
    }
  }
}
)mlir";

  MLIRJitModule M(Target, Code);
  auto Write42 = M.getKernel<void(int *)>("write42");

  int *DeviceBuffer = nullptr;
  gpuErrCheck(gpuMalloc(reinterpret_cast<void **>(&DeviceBuffer), sizeof(int)));

  int Initial = 0;
  gpuErrCheck(
      gpuMemcpy(DeviceBuffer, &Initial, sizeof(int), gpuMemcpyHostToDevice));

  gpuErrCheck(Write42.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr, DeviceBuffer));
  gpuErrCheck(gpuDeviceSynchronize());

  int Result = -1;
  gpuErrCheck(
      gpuMemcpy(&Result, DeviceBuffer, sizeof(int), gpuMemcpyDeviceToHost));
  std::printf("mlir_gpu_jit_module_source: Result=%d\n", Result);
  assert(Result == 42 && "MLIRJitModule GPU execution returned wrong result");

  gpuErrCheck(gpuFree(DeviceBuffer));
  return 0;
}

// clang-format off
// CHECK: mlir_gpu_jit_module_source: Result=42
// CHECK-FIRST: [proteus][Dispatcher{{CUDA|HIP}}] StorageCache rank 0 hits 0 accesses 1
// CHECK-SECOND: [proteus][Dispatcher{{CUDA|HIP}}] StorageCache rank 0 hits 1 accesses 1
// clang-format on
