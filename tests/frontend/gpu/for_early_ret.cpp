// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/for_early_ret.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/for_early_ret.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <iostream>

#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>

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

  // Void early-terminate inside loop after i == 5.
  auto KHVoid = J.addKernel<void(double*)>("for_early_void");
  auto &FV = KHVoid.F;
  {
    auto &Arr = FV.getArg(0);
    auto &I = FV.declVar<int>("i");
    auto &Inc = FV.declVar<int>("inc");
    auto &UB = FV.declVar<int>("ub");

    FV.beginFunction();
    {
      I = 0;
      UB = 10;
      Inc = 1;
      FV.beginFor(I, I, UB, Inc);
      {
        Arr[I] = Arr[I] + 1.0;
        FV.beginIf(I == 5);
        { FV.ret(); }
        FV.endIf();
      }
      FV.endFor();
      FV.ret();
    }
    FV.endFunction();
  }


  J.compile();

  double *X;
  gpuErrCheck(gpuMallocManaged(&X, sizeof(double) * 10));
  for (int I = 0; I < 10; I++) {
    X[I] = 1.0;
  }

  gpuErrCheck(KHVoid.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr, X));
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
// CHECK-NEXT: X[6] = 1
// CHECK-NEXT: X[7] = 1
// CHECK-NEXT: X[8] = 1
// CHECK-NEXT: X[9] = 1
// CHECK: JitCache hits 0 total 2
// CHECK-COUNT-2: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
