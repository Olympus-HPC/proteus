// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/if_early_ret.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/if_early_ret.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
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

  // Void with early-return in IF guarding a side effect.
  auto KHVoid = J.addKernel<void(double*, double, double)>("if_early_void");
  auto &G = KHVoid.F;
  {
    auto &Arr = G.getArg(0);
    auto &A = G.getArg(1);
    auto &B = G.getArg(2);

    G.beginFunction();
    {
      Arr[0] = 0.0;
      G.beginIf(A < B);
      { G.ret(); }
      G.endIf();
      Arr[0] = Arr[0] + 1.0;
      G.ret();
    }
    G.endFunction();
  }

  J.compile();

  // Void checks
  double *X;
  gpuErrCheck(gpuMallocManaged(&X, sizeof(double)));
  X[0] = 0.0;
  gpuErrCheck(KHVoid.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr, X, 1.0, 2.0));
  gpuErrCheck(gpuDeviceSynchronize());
  std::cout << "X " << X[0] << "\n";
  X[0] = 0.0;
  gpuErrCheck(KHVoid.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr, X, 2.0, 1.0));
  gpuErrCheck(gpuDeviceSynchronize());
  std::cout << "X " << X[0] << "\n";

  gpuErrCheck(gpuFree(X));

  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK: X 0
// CHECK-NEXT: X 1
// CHECK: JitCache hits 0 total 1
// CHECK-COUNT-1: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
