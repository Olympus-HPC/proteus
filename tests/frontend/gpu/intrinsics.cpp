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
  proteus::init();

  auto J = JitModule(TARGET);
  auto KernelHandle = J.addKernel<void(float*, float*, float*, float*, float*, float*)>(
      "intrinsics");
  auto &F = KernelHandle.F;

  auto &PowOut = F.getArg<0>();
  auto &SqrtOut = F.getArg<1>();
  auto &ExpOut = F.getArg<2>();
  auto &SinOut = F.getArg<3>();
  auto &CosOut = F.getArg<4>();
  auto &FabsOut = F.getArg<5>();

  F.beginFunction();
  {
    // powf(float, int)
    auto B = F.declVar<float>();
    auto E = F.declVar<int>();
    B = 2.0f;
    E = 3;
    PowOut[0] = powf(B, E);

    // sqrtf(int)
    auto X = F.declVar<int>();
    X = 9;
    SqrtOut[0] = sqrtf(X);

    // expf(int)
    auto Y = F.declVar<int>();
    Y = 1;
    ExpOut[0] = expf(Y);

    // sinf(0.0f) and cosf(0.0f)
    auto Z = F.declVar<float>();
    Z = 0.0f;
    SinOut[0] = sinf(Z);
    CosOut[0] = cosf(Z);

    // fabs(int)
    auto W = F.declVar<int>();
    W = -5;
    FabsOut[0] = fabs(W);

    F.ret();
  }
  F.endFunction();

  J.compile();

  float *Pow, *Sqrt, *Exp, *Sin, *Cos, *Fabs;
  gpuErrCheck(gpuMallocManaged(&Pow, sizeof(float)));
  gpuErrCheck(gpuMallocManaged(&Sqrt, sizeof(float)));
  gpuErrCheck(gpuMallocManaged(&Exp, sizeof(float)));
  gpuErrCheck(gpuMallocManaged(&Sin, sizeof(float)));
  gpuErrCheck(gpuMallocManaged(&Cos, sizeof(float)));
  gpuErrCheck(gpuMallocManaged(&Fabs, sizeof(float)));

  gpuErrCheck(KernelHandle.launch({1,1,1},{1,1,1},0,nullptr,
                                  Pow, Sqrt, Exp, Sin, Cos, Fabs));
  gpuErrCheck(gpuDeviceSynchronize());

  std::cout << "pow = " << *Pow << "\n";
  std::cout << "sqrt = " << *Sqrt << "\n";
  std::cout << "exp = " << *Exp << "\n";
  std::cout << "sin = " << *Sin << "\n";
  std::cout << "cos = " << *Cos << "\n";
  std::cout << "fabs = " << *Fabs << "\n";

  gpuErrCheck(gpuFree(Pow));
  gpuErrCheck(gpuFree(Sqrt));
  gpuErrCheck(gpuFree(Exp));
  gpuErrCheck(gpuFree(Sin));
  gpuErrCheck(gpuFree(Cos));
  gpuErrCheck(gpuFree(Fabs));

  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK: pow = 8
// CHECK-NEXT: sqrt = 3
// CHECK-NEXT: exp = 2.71828
// CHECK-NEXT: sin = 0
// CHECK-NEXT: cos = 1
// CHECK-NEXT: fabs = 5
// CHECK-FIRST: JitCache hits 0 total 1
// CHECK: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
// clang-format on
