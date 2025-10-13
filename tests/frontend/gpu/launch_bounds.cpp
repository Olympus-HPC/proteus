// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/launch_bounds.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-%ext
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <proteus/JitFrontend.hpp>

using namespace proteus;

#if PROTEUS_ENABLE_HIP
#define TARGET "hip"
constexpr int MaxThreadsPerBlock = 256;
#elif PROTEUS_ENABLE_CUDA
#define TARGET "cuda"
constexpr int MaxThreadsPerBlock = 128;
#else
#error "Expected PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA defined"
#endif
constexpr int MinBlocksPerSM = 4;

int main() {
  auto J = proteus::JitModule(TARGET);

  auto KernelHandle = J.addKernelTT<void()>("dsl_static_bounds");
  auto &F = KernelHandle.F;

  F.beginFunction();
  { F.ret(); }
  F.endFunction();

  KernelHandle.setLaunchBounds(MaxThreadsPerBlock, MinBlocksPerSM);

  J.print();
  return 0;
}

// CHECK: define{{.*}}@dsl_static_bounds(
// CHECK-HIP: "amdgpu-flat-work-group-size"="1,256"
// CHECK-CUDA: !{ptr @dsl_static_bounds, !"maxntid", i32 128}
// CHECK-CUDA: !{ptr @dsl_static_bounds, !"minctasm", i32 4}
