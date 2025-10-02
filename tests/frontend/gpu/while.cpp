// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/while.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/while.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <iostream>

#include <proteus/Frontend/Builtins.hpp>
#include <proteus/JitFrontend.hpp>
#include <proteus/JitInterface.hpp>
#include <proteus/Utils.h>

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
  proteus::init();

  auto J = proteus::JitModule(TARGET);
  auto KernelHandle = J.addKernel<void(double *, int)>("while");
  auto &F = KernelHandle.F;

  auto &A = F.getArg(0);
  auto &N = F.getArg(1);
  auto &I = F.declVar<int>("i");
  auto &Stride = F.declVar<int>("stride");

  F.beginFunction();
  {
    I = F.callBuiltin(getBlockIdX) * F.callBuiltin(getBlockDimX) +
        F.callBuiltin(getThreadIdX);
    Stride = F.callBuiltin(getGridDimX) * F.callBuiltin(getBlockDimX);
    auto &Cond = F.declVar<bool>("cond");
    Cond = I < N;
    F.beginWhile(Cond);
    {
      A[I] = A[I] + I;
      I = I + Stride;
      Cond = I < N;
    }
    F.endWhile();
    F.ret();
  }
  F.endFunction();

  J.compile();

  double *X;
  int NHost = 10;
  gpuErrCheck(gpuMallocManaged(&X, sizeof(double) * NHost));
  for (int I = 0; I < NHost; I++)
    X[I] = 1.0;

  gpuErrCheck(KernelHandle.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr, X, NHost));
  gpuErrCheck(gpuDeviceSynchronize());

  for (int I = 0; I < NHost; I++)
    std::cout << "X[" << I << "] = " << X[I] << "\n";

  gpuErrCheck(gpuFree(X));

  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK: X[0] = 1
// CHECK-NEXT: X[1] = 2
// CHECK-NEXT: X[2] = 3
// CHECK-NEXT: X[3] = 4
// CHECK-NEXT: X[4] = 5
// CHECK-NEXT: X[5] = 6
// CHECK-NEXT: X[6] = 7
// CHECK-NEXT: X[7] = 8
// CHECK-NEXT: X[8] = 9
// CHECK-NEXT: X[9] = 10
// CHECK: JitCache hits 0 total 1
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
