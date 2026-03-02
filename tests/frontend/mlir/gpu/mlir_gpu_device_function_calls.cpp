// clang-format off
// RUN: %build/mlir_gpu_device_function_calls.%ext | %FILECHECK %s
// clang-format on

#include <memory>
#include <proteus/Frontend/Builtins.h>
#include <proteus/JitFrontend.h>

using namespace proteus;

#if PROTEUS_ENABLE_HIP
#define TARGET "hip"
#elif PROTEUS_ENABLE_CUDA
#define TARGET "cuda"
#else
#error "Expected PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA defined"
#endif

int main() {
  auto J = std::make_unique<JitModule>(TARGET, "mlir");

  auto &Helper = J->addFunction<int()>("device_helper");
  auto KernelHandle = J->addKernel<void()>("kernel_calls_helper");
  auto &K = KernelHandle.F;

  Helper.beginFunction();
  {
    // Builtins are valid in any gpu.func in device mode.
    auto Tx = Helper.callBuiltin(builtins::gpu::getThreadIdX);
    Helper.ret(Tx);
  }
  Helper.endFunction();

  K.beginFunction();
  {
    auto Ret = K.call<int()>("device_helper");
    auto Sink = K.declVar<int>("sink");
    Sink = Ret;
    K.ret();
  }
  K.endFunction();

  J->print();
  return 0;
}

// clang-format off
// CHECK-LABEL: gpu.module @kernels
// CHECK: func.func private @device_helper() -> i32
// CHECK: gpu.thread_id  x
// CHECK-NOT: kernel
// CHECK-LABEL: gpu.func @kernel_calls_helper() kernel
// CHECK: func.call @device_helper() : () -> i32
// clang-format on
