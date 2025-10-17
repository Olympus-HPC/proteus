// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/get_address.%ext | %FILECHECK %s --check-prefixes=CHECK
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <iostream>

#include <proteus/Frontend/Builtins.hpp>
#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

#include "../../gpu/gpu_common.h"

using namespace proteus;
using namespace builtins::gpu;
#if PROTEUS_ENABLE_HIP
#define TARGET "hip"
#elif PROTEUS_ENABLE_CUDA
#define TARGET "cuda"
#else
#error "Expected PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA defined"
#endif

int main() {
  auto J = proteus::JitModule(TARGET);

  auto KernelHandle =
      J.addKernel<void(int *, double *)>("test_get_address");
  auto &F = KernelHandle.F;

  F.beginFunction();
  {
    auto &IntResult = F.getArg<0>();
    auto &DoubleResult = F.getArg<1>();

    auto Tid = F.callBuiltin(getThreadIdX);
    auto Bid = F.callBuiltin(getBlockIdX);
    auto BlockDim = F.callBuiltin(getBlockDimX);

    auto Idx = F.declVar<int>();
    Idx = Bid * BlockDim + Tid;

    F.beginIf(Idx == 0);
    {
      auto IntVar = F.declVar<int>("int_var");
      IntVar = 42;

      auto IntAddr = IntVar.getAddress();
      IntResult[0] = IntAddr[0];

      auto DoubleVar = F.declVar<double>("double_var");
      DoubleVar = 3.14159;

      auto DoubleAddr = DoubleVar.getAddress();
      DoubleResult[0] = DoubleAddr[0];

      IntAddr[0] = 100;
      IntResult[1] = IntVar;

      DoubleAddr[0] = 2.71828;
      DoubleResult[1] = DoubleVar;
    }
    F.endIf();

    F.ret();
  }
  F.endFunction();

  int *IntResults;
  double *DoubleResults;

  gpuErrCheck(gpuMallocManaged(&IntResults, 2 * sizeof(int)));
  gpuErrCheck(gpuMallocManaged(&DoubleResults, 2 * sizeof(double)));

  J.compile();

  std::cout << "Launching kernel...\n";
  gpuErrCheck(KernelHandle.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr, IntResults,
                                  DoubleResults));

  gpuErrCheck(gpuDeviceSynchronize());

  std::cout << "ScalarVar<int> getAddress test:\n";
  std::cout << "  Original value: " << IntResults[0] << "\n";
  std::cout << "  Modified value: " << IntResults[1] << "\n";

  std::cout << "ScalarVar<double> getAddress test:\n";
  std::cout << "  Original value: " << DoubleResults[0] << "\n";
  std::cout << "  Modified value: " << DoubleResults[1] << "\n";

  bool passed = (IntResults[0] == 42) && (IntResults[1] == 100) &&
                (DoubleResults[0] > 3.14 && DoubleResults[0] < 3.15) &&
                (DoubleResults[1] > 2.71 && DoubleResults[1] < 2.72);

  if (passed) {
    std::cout << "All tests passed!\n";
  } else {
    std::cout << "Some tests failed!\n";
  }

  gpuErrCheck(gpuFree(IntResults));
  gpuErrCheck(gpuFree(DoubleResults));

  return passed ? 0 : 1;
}

// clang-format off
// CHECK: ScalarVar<int> getAddress test:
// CHECK-NEXT:   Original value: 42
// CHECK-NEXT:   Modified value: 100
// CHECK: ScalarVar<double> getAddress test:
// CHECK-NEXT:   Original value: 3.14159
// CHECK-NEXT:   Modified value: 2.71828
// CHECK: All tests passed!
