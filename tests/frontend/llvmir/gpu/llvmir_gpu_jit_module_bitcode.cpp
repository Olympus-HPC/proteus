// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: %llvm_as %S/llvmir_gpu_jit_module_bitcode_input.%ext.ll -o %t.module.bc
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/llvmir_gpu_jit_module_bitcode.%ext %t.module.bc | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/llvmir_gpu_jit_module_bitcode.%ext %t.module.bc | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <cassert>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>

#include <proteus/LLVMIRJitModule.h>

#include "../../../gpu/gpu_common.h"

using namespace proteus;

int main(int argc, char **argv) {
#if defined(PROTEUS_ENABLE_CUDA)
  const char *Target = "cuda";
#elif defined(PROTEUS_ENABLE_HIP)
  const char *Target = "hip";
#else
  return 0;
#endif

  if (argc != 2) {
    std::fprintf(stderr, "Expected a single bitcode input path\n");
    return 1;
  }

  std::ifstream Input(argv[1], std::ios::binary);
  assert(Input && "Failed to open bitcode file");

  std::string Code((std::istreambuf_iterator<char>(Input)),
                   std::istreambuf_iterator<char>());

  LLVMIRJitModule M(Target, Code, LLVMIRInputKind::Bitcode);
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
  std::printf("llvmir_gpu_jit_module_bitcode: Result=%d\n", Result);
  assert(Result == 42 &&
         "LLVMIRJitModule GPU bitcode execution returned wrong result");

  gpuErrCheck(gpuFree(DeviceBuffer));
  return 0;
}

// clang-format off
// CHECK: llvmir_gpu_jit_module_bitcode: Result=42
// CHECK-FIRST: [proteus][Dispatcher{{CUDA|HIP}}] StorageCache rank 0 hits 0 accesses 1
// CHECK-SECOND: [proteus][Dispatcher{{CUDA|HIP}}] StorageCache rank 0 hits 1 accesses 1
// clang-format on
