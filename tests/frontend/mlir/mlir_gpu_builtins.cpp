// clang-format off
// RUN: %build/mlir_gpu_builtins | %FILECHECK %s
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
  auto &K = J->addFunction<void()>("kernel");

  K.beginFunction();
  {
    K.setKernel();

    auto TX = K.callBuiltin(builtins::gpu::getThreadIdX);
    auto BX = K.callBuiltin(builtins::gpu::getBlockIdX);
    auto BDX = K.callBuiltin(builtins::gpu::getBlockDimX);
    auto GDX = K.callBuiltin(builtins::gpu::getGridDimX);

    auto Sink = TX + BX + BDX + GDX;
    (void)Sink;

    K.callBuiltin(builtins::gpu::syncThreads);
    K.ret();
  }
  K.endFunction();

  J->print();
  return 0;
}

// clang-format off
// CHECK: func.func @kernel() attributes {gpu.kernel}
// CHECK: gpu.thread_id x
// CHECK: gpu.block_id x
// CHECK: gpu.block_dim x
// CHECK: gpu.grid_dim x
// CHECK: gpu.barrier
// clang-format on
