#include <cassert>
#include <cstdio>

#include <proteus/Frontend/Builtins.h>
#include <proteus/JitFrontend.h>

#include "../../gpu/gpu_common.h"

using namespace proteus;

int main() {
#if defined(PROTEUS_ENABLE_CUDA)
  const char *Target = "cuda";
#elif defined(PROTEUS_ENABLE_HIP)
  const char *Target = "hip";
#else
  return 0;
#endif

  // This test validates end-to-end GPU execution for MLIR backend:
  // MLIR gpu dialect -> (NVVM/ROCDL) -> LLVM IR, then Proteus device
  // dispatcher compilation, module loading, and kernel launch.
  JitModule J(Target, "mlir");
  auto KernelHandle = J.addKernel<void(int *)>("mlir_gpu_kernel_execute");
  auto &K = KernelHandle.F;

  K.beginFunction();
  {
    auto TidX = K.callBuiltin(builtins::gpu::getThreadIdX);
    K.beginIf(TidX == 0);
    {
      auto &Buffer = K.getArg<0>();
      Buffer[0] = 42;
    }
    K.endIf();
    K.ret();
  }
  K.endFunction();

  J.print();
  J.printLLVMIR();

  int *DeviceBuffer = nullptr;
  gpuErrCheck(gpuMalloc(reinterpret_cast<void **>(&DeviceBuffer), sizeof(int)));

  int Initial = 0;
  gpuErrCheck(
      gpuMemcpy(DeviceBuffer, &Initial, sizeof(int), gpuMemcpyHostToDevice));

  gpuErrCheck(
      KernelHandle.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr, DeviceBuffer));
  gpuErrCheck(gpuDeviceSynchronize());

  int Result = -1;
  gpuErrCheck(
      gpuMemcpy(&Result, DeviceBuffer, sizeof(int), gpuMemcpyDeviceToHost));
  std::printf("mlir_gpu_kernel_execute: Result=%d\n", Result);
  assert(Result == 42 && "MLIR GPU kernel execution produced wrong result");

  gpuErrCheck(gpuFree(DeviceBuffer));
  return 0;
}
