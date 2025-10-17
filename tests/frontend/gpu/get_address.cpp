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

  auto KernelHandle = J.addKernel<void(int *, double *)>("test_get_address");
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

      auto IntAddr = IntVar.getAddress();
      IntAddr[0] = 42;
      IntResult[0] = IntAddr[0];

      auto DoubleVar = F.declVar<double>("double_var");
      DoubleVar = 3.14159;

      auto DoubleAddr = DoubleVar.getAddress();
      DoubleResult[0] = DoubleAddr[0];

      IntAddr[0] = 100;
      IntResult[1] = IntVar;

      DoubleAddr[0] = 2.71828;
      DoubleResult[1] = DoubleVar;

      auto IntVar2 = F.declVar<int>("int_var2");
      IntVar2 = 7;
      auto P = IntVar2.getAddress();
      auto PP = P.getAddress();
      IntResult[2] = PP[0][0];
      PP[0][0] = 123;
      IntResult[3] = IntVar2;
    }
    F.endIf();

    F.ret();
  }
  F.endFunction();

  int *IntResults;
  double *DoubleResults;

  gpuErrCheck(gpuMallocManaged(&IntResults, 4 * sizeof(int)));
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

  std::cout << "PointerVar<int*> getAddress test:\n";
  std::cout << "  Original through **: " << IntResults[2] << "\n";
  std::cout << "  Modified via **: " << IntResults[3] << "\n";

  bool passed = (IntResults[0] == 42) && (IntResults[1] == 100) &&
                (DoubleResults[0] > 3.14 && DoubleResults[0] < 3.15) &&
                (DoubleResults[1] > 2.71 && DoubleResults[1] < 2.72) &&
                (IntResults[2] == 7) && (IntResults[3] == 123);

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
// CHECK: PointerVar<int*> getAddress test:
// CHECK-NEXT:   Original through **: 7
// CHECK-NEXT:   Modified via **: 123
// CHECK: All tests passed!
