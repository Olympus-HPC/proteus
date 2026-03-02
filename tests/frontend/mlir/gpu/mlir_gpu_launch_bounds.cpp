// clang-format off
// RUN: %build/mlir_gpu_launch_bounds.%ext | %FILECHECK %s
// clang-format on

#include <memory>
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

  auto KernelHandle = J->addKernel<void()>("kernel_lb");
  auto &K = KernelHandle.F;

  K.beginFunction();
  { K.ret(); }
  K.endFunction();

  KernelHandle.setLaunchBounds(256, 2);

  J->print();
  return 0;
}

// clang-format off
// CHECK: {{(\"gpu\.module\"\(\)|gpu\.module)}}{{.*(@kernels|sym_name = \"kernels\")}}
// CHECK: {{(\"gpu\.func\"\(\)|gpu\.func)}}{{.*(@kernel_lb|sym_name = \"kernel_lb\")}}{{.* kernel}}
// CHECK-DAG: {{(nvvm\.maxntid|rocdl\.flat_work_group_size)}}
// CHECK-DAG: {{(nvvm\.minctasm|rocdl\.waves_per_eu)}}
// clang-format on
