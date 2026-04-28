// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/llvmir_gpu_jit_module_source.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/llvmir_gpu_jit_module_source.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cassert>
#include <cstdio>

#include <proteus/LLVMIRJitModule.h>

#include "../../../gpu/gpu_common.h"

using namespace proteus;

int main() {
#if defined(PROTEUS_ENABLE_CUDA)
  const char *Target = "cuda";
  static constexpr const char *Code = R"llvm(
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @write42(ptr addrspace(1) %out) {
entry:
  store i32 42, ptr addrspace(1) %out, align 4
  ret void
}

!nvvm.annotations = !{!0}
!0 = !{ptr @write42, !"kernel", i32 1}
)llvm";
#elif defined(PROTEUS_ENABLE_HIP)
  const char *Target = "hip";
  static constexpr const char *Code = R"llvm(
target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @write42(ptr addrspace(1) %out) #0 {
entry:
  store i32 42, ptr addrspace(1) %out, align 4
  ret void
}

attributes #0 = { nounwind }
)llvm";
#else
  return 0;
#endif

  LLVMIRJitModule M(Target, Code);
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
  std::printf("llvmir_gpu_jit_module_source: Result=%d\n", Result);
  assert(Result == 42 && "LLVMIRJitModule GPU execution returned wrong result");

  gpuErrCheck(gpuFree(DeviceBuffer));
  return 0;
}

// clang-format off
// CHECK: llvmir_gpu_jit_module_source: Result=42
// CHECK-FIRST: [proteus][Dispatcher{{CUDA|HIP}}] StorageCache rank 0 hits 0 accesses 1
// CHECK-SECOND: [proteus][Dispatcher{{CUDA|HIP}}] StorageCache rank 0 hits 1 accesses 1
// clang-format on
