// clang-format off
// RUN: %build/mlir_gpu_kernel_representation | %FILECHECK %s
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

  // Kernel-ness is declared at construction time via addKernel.
  auto KernelHandle = J->addKernel<void()>("kernel_repr");
  auto &K = KernelHandle.F;

  K.beginFunction();
  {
    auto TX = K.callBuiltin(builtins::gpu::getThreadIdX);
    (void)TX;
    K.ret();
  }
  K.endFunction();

  J->print();
  return 0;
}

// clang-format off
// CHECK: "gpu.module"()
// CHECK: sym_name = "kernels"
// CHECK: "gpu.func"()
// CHECK: "gpu.thread_id"()
// CHECK: gpu.kernel
// CHECK: sym_name = "kernel_repr"
// clang-format on
