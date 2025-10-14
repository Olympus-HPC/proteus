// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/multi_module.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/multi_module.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <proteus/Frontend/Builtins.hpp>
#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

#include "../../gpu/gpu_common.h"

using namespace proteus;

#if PROTEUS_ENABLE_HIP
#define TARGET "hip"

#elif PROTEUS_ENABLE_CUDA
#define TARGET "cuda"

#else
#error "Expected PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA defined"
#endif

auto createJitModule1() {
  auto J = std::make_unique<JitModule>(TARGET);

  auto KernelHandle1 = J->addKernel<void(double *)>("kernel1");
  {
    auto &F = KernelHandle1.F;

    F.beginFunction();
    {
      auto [V] = F.getArgs();
      *V = 42;

      F.ret();
    }
    F.endFunction();
  }

  auto KernelHandle2 = J->addKernel<void(double *)>("kernel2");
  {
    auto &F = KernelHandle2.F;

    F.beginFunction();
    {
      auto [V] = F.getArgs();
      *V = 23;

      F.ret();
    }
    F.endFunction();
  }

  return std::make_tuple(std::move(J), KernelHandle1, KernelHandle2);
}

auto createJitModule2() {
  auto J = std::make_unique<JitModule>(TARGET);

  auto KernelHandle1 = J->addKernel<void(double *)>("kernel1");
  {
    auto &F = KernelHandle1.F;

    F.beginFunction();
    {
      auto [V] = F.getArgs();
      *V = 142;

      F.ret();
    }
    F.endFunction();
  }

  auto KernelHandle2 = J->addKernel<void(double *)>("kernel2");
  {
    auto &F = KernelHandle2.F;

    F.beginFunction();
    {
      auto [V] = F.getArgs();
      *V = 123;

      F.ret();
    }
    F.endFunction();
  }

  return std::make_tuple(std::move(J), KernelHandle1, KernelHandle2);
}

int main() {
  auto [J1, KernelHandle11, KernelHandle12] = createJitModule1();
  auto [J2, KernelHandle21, KernelHandle22] = createJitModule2();

  double *V;
  gpuErrCheck(gpuMallocManaged(&V, sizeof(double)));

  gpuErrCheck(KernelHandle11.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr, V));
  gpuErrCheck(gpuDeviceSynchronize());
  std::cout << "V " << *V << "\n";

  gpuErrCheck(KernelHandle12.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr, V));
  gpuErrCheck(gpuDeviceSynchronize());
  std::cout << "V " << *V << "\n";

  gpuErrCheck(KernelHandle21.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr, V));
  gpuErrCheck(gpuDeviceSynchronize());
  std::cout << "V " << *V << "\n";

  gpuErrCheck(KernelHandle22.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr, V));
  gpuErrCheck(gpuDeviceSynchronize());
  std::cout << "V " << *V << "\n";

  gpuErrCheck(gpuFree(V));

  return 0;
}

// clang-format off
// CHECK: V 42
// CHECK: V 23
// CHECK: V 142
// CHECK: V 123
// CHECK: JitCache hits 0 total 4
// CHECK-COUNT-4: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 2
// CHECK-SECOND: JitStorageCache hits 2 total 2
