// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/intrinsics.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/intrinsics.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
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

using namespace proteus;

int main() {

  auto J = JitModule(TARGET);
  auto KernelHandle =
      J.addKernel<void(float *, float *, float *, float *)>("intrinsics");
  auto &F = KernelHandle.F;
  auto &Arg0 = F.getArg(0);
  auto &Arg1 = F.getArg(1);
  auto &Arg2 = F.getArg(2);
auto &Arg3 = F.getArg(3);
  F.beginFunction();
  {
    auto &X = F.declVar<float>();
    X = 2.0f;
    auto &Y = F.declVar<float>();
    Y = 3.0f;

    auto &Z = F.declVar<float>();
    Z = 9.0f;

    auto &W = F.declVar<float>();
    W = 0.0f;

    auto &A = F.declVar<float>();
    A = 1.0f;

    Arg0[0] = powf(X, Y);   // 8
    Arg1[0] = sqrtf(Z);     // 3
    Arg2[0] = expf(W);      // 1
    Arg3[0] = logf(A);      // 0

    F.ret();
  }
  F.endFunction();

  J.compile();

  float *R0, *R1, *R2, *R3;
  gpuErrCheck(gpuMallocManaged(&R0, sizeof(float)));
  gpuErrCheck(gpuMallocManaged(&R1, sizeof(float)));
  gpuErrCheck(gpuMallocManaged(&R2, sizeof(float)));
   gpuErrCheck(gpuMallocManaged(&R3, sizeof(float)));
  
  gpuErrCheck(KernelHandle.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr, R0, R1, R2, R3));
   gpuErrCheck(gpuDeviceSynchronize());
  
  std::cout << "R0 = " << *R0 << "\n";
  std::cout << "R1 = " << *R1 << "\n";
   std::cout << "R2 = " << *R2 << "\n";
  std::cout << "R3 = " << *R3 << "\n";
  
  gpuErrCheck(gpuFree(R0));
  gpuErrCheck(gpuFree(R1));
  gpuErrCheck(gpuFree(R2));
  gpuErrCheck(gpuFree(R3));

  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK: R0 = 8
// CHECK-NEXT: R1 = 3
// CHECK-NEXT: R2 = 1
// CHECK-NEXT: R3 = 0
// CHECK-NEXT: JitCache hits 0 total 1
// CHECK-NEXT: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
// clang-format on
