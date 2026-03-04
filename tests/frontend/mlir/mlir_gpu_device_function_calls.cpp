// clang-format off
// RUN: %build/mlir_gpu_device_function_calls | %FILECHECK %s
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
// CHECK: "gpu.module"()
// CHECK: sym_name = "kernels"
// CHECK: "gpu.func"() <{function_type = () -> i32}>
// CHECK: "gpu.thread_id"()
// CHECK: sym_name = "device_helper"
// CHECK-NOT: gpu.kernel
// CHECK: "gpu.func"() <{function_type = () -> ()}>
// CHECK: "func.call"() <{callee = @device_helper
// CHECK: gpu.kernel
// CHECK: sym_name = "kernel_calls_helper"
// clang-format on
