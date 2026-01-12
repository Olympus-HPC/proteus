// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/get_address.%ext | %FILECHECK %s --check-prefixes=CHECK
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <proteus/Frontend/Builtins.h>
#include <proteus/JitFrontend.h>

#include "../../gpu/gpu_common.h"

#include <iostream>

#if PROTEUS_ENABLE_HIP
#define TARGET "hip"
#elif PROTEUS_ENABLE_CUDA
#define TARGET "cuda"
#else
#error "Expected PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA defined"
#endif

using namespace proteus;
using namespace builtins::gpu;

int main() {
  auto J = proteus::JitModule(TARGET);

  {
    auto &Setter = J.addFunction<void(int *)>("set_via_ptr");
    Setter.beginFunction();
    {
      auto &P = Setter.getArg<0>();
      P[0] = 111;
      Setter.ret();
    }
    Setter.endFunction();
  }

  auto KernelHandle = J.addKernel<void(int *)>("test_get_address");
  auto &F = KernelHandle.F;

  F.beginFunction();
  {
    auto &Out = F.getArg<0>();

    auto Tid = F.callBuiltin(getThreadIdX);
    auto Bid = F.callBuiltin(getBlockIdX);
    auto BlockDim = F.callBuiltin(getBlockDimX);
    auto Idx = F.declVar<int>();
    Idx = Bid * BlockDim + Tid;

    F.beginIf(Idx == 0);
    {
      auto Scalar = F.declVar<int>("scalar");
      Scalar = 7;

      auto ScalarAddr = Scalar.getAddress();
      Out[0] = ScalarAddr[0];

      F.call<void(int *)>("set_via_ptr", ScalarAddr);
      Out[3] = Scalar;

      auto P = Out.getAddress();
      Out[1] = P[0][0];
      P[0][0] = 222;
      Out[2] = Out[0];
    }
    F.endIf();

    F.ret();
  }
  F.endFunction();

  int *Results;
  gpuErrCheck(gpuMallocManaged(&Results, 4 * sizeof(int)));

  J.compile();

  gpuErrCheck(KernelHandle.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr, Results));
  gpuErrCheck(gpuDeviceSynchronize());

  std::cout << "GPU getAddress test:\n";
  std::cout << "  scalar via *: " << Results[0] << "\n";
  std::cout << "  pointer via **: " << Results[1] << "\n";
  std::cout << "  after store via **: " << Results[2] << "\n";
  std::cout << "  after internal set_via_ptr: " << Results[3] << "\n";

  bool Ok = (Results[0] == 222) && (Results[1] == 7) && (Results[2] == 222) &&
            (Results[3] == 111);

  gpuErrCheck(gpuFree(Results));

  return Ok ? 0 : 1;
}

// clang-format off
// CHECK: GPU getAddress test:
// CHECK-NEXT:   scalar via *: 222
// CHECK-NEXT:   pointer via **: 7
// CHECK-NEXT:   after store via **: 222
// CHECK-NEXT:   after internal set_via_ptr: 111
