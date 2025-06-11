// RUN: rm -rf .proteus
// RUN: ./for.%ext | %FILECHECK %s --check-prefixes=CHECK
// RUN: rm -rf .proteus

#include <iostream>

#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>
#include <proteus/Utils.h>

#include "../../gpu/gpu_common.h"

#if PROTEUS_ENABLE_HIP
#define TARGET "hip"
#elif PROTEUS_ENABLE_CUDA
#define TARGET "cuda"
#else
#error "Expected PROTEUS_ENABLE_HIP or PROTEUS_ENABLE_CUDA defined"
#endif

int main() {
  proteus::init();

  auto J = proteus::JitModule(TARGET);
  auto KernelHandle = J.addKernel<double *>("for");
  auto &F = KernelHandle.F;

  auto &I = F.declVar<int>("i");
  auto &Inc = F.declVar<int>("inc");
  auto &UB = F.declVar<int>("ub");
  auto &Arg = F.getArg(0);
  F.beginFunction();
  {
    I = 0;
    UB = 10;
    Inc = 1;
    F.beginFor(I, I, UB, Inc);
    { Arg[I] = Arg[I] + 1.0; }
    F.endFor();
    F.ret();
  }
  F.endFunction();

  J.compile();

  double *X;
  gpuErrCheck(gpuMallocManaged(&X, sizeof(double) * 10));
  for (int I = 0; I < 10; I++) {
    X[I] = 1.0;
  }

  gpuErrCheck(KernelHandle.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr, X));
  gpuErrCheck(gpuDeviceSynchronize());

  for (int I = 0; I < 10; I++)
    std::cout << "X[" << I << "] = " << X[I] << "\n";

  gpuErrCheck(gpuFree(X));

  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK: X[0] = 2
// CHECK-NEXT: X[1] = 2
// CHECK-NEXT: X[2] = 2
// CHECK-NEXT: X[3] = 2
// CHECK-NEXT: X[4] = 2
// CHECK-NEXT: X[5] = 2
// CHECK-NEXT: X[6] = 2
// CHECK-NEXT: X[7] = 2
// CHECK-NEXT: X[8] = 2
// CHECK-NEXT: X[9] = 2
