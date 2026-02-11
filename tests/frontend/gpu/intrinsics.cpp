// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/intrinsics.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/intrinsics.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <iostream>

#include <proteus/Frontend/Builtins.h>
#include <proteus/JitFrontend.h>

#include "../../gpu/gpu_common.h"

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
  auto J = JitModule(TARGET);
  auto KernelHandle =
      J.addKernel<void(float *, float *, float *, float *, float *, float *,
                       float *, int *, float *, float *, float *, int *)>(
          "intrinsics");
  auto &F = KernelHandle.F;

  auto &PowOut = F.getArg<0>();
  auto &SqrtOut = F.getArg<1>();
  auto &ExpOut = F.getArg<2>();
  auto &SinOut = F.getArg<3>();
  auto &CosOut = F.getArg<4>();
  auto &FabsOut = F.getArg<5>();
  auto &PowBase = F.getArg<6>();
  auto &PowExp = F.getArg<7>();
  auto &SqrtIn = F.getArg<8>();
  auto &ExpIn = F.getArg<9>();
  auto &TrigIn = F.getArg<10>();
  auto &FabsIn = F.getArg<11>();

  F.beginFunction();
  {
    auto Tid = F.callBuiltin(getThreadIdX);
    PowOut[Tid] = powf(PowBase[Tid], PowExp[Tid]);
    SqrtOut[Tid] = sqrtf(SqrtIn[Tid]);
    ExpOut[Tid] = expf(ExpIn[Tid]);
    SinOut[Tid] = sinf(TrigIn[Tid]);
    CosOut[Tid] = cosf(TrigIn[Tid]);
    FabsOut[Tid] = fabs(FabsIn[Tid]);

    F.ret();
  }
  F.endFunction();

  J.compile();

  float *Pow, *Sqrt, *Exp, *Sin, *Cos, *Fabs;
  float *PowBaseHost, *ExpInHost, *TrigHost, *SqrtHost;
  int *PowExpHost, *FabsHost;
  gpuErrCheck(gpuMallocManaged(&Pow, sizeof(float)));
  gpuErrCheck(gpuMallocManaged(&Sqrt, sizeof(float)));
  gpuErrCheck(gpuMallocManaged(&Exp, sizeof(float)));
  gpuErrCheck(gpuMallocManaged(&Sin, sizeof(float)));
  gpuErrCheck(gpuMallocManaged(&Cos, sizeof(float)));
  gpuErrCheck(gpuMallocManaged(&Fabs, sizeof(float)));
  gpuErrCheck(gpuMallocManaged(&PowBaseHost, sizeof(float)));
  gpuErrCheck(gpuMallocManaged(&PowExpHost, sizeof(int)));
  gpuErrCheck(gpuMallocManaged(&SqrtHost, sizeof(float)));
  gpuErrCheck(gpuMallocManaged(&ExpInHost, sizeof(float)));
  gpuErrCheck(gpuMallocManaged(&TrigHost, sizeof(float)));
  gpuErrCheck(gpuMallocManaged(&FabsHost, sizeof(int)));

  *PowBaseHost = 1.5f;
  *PowExpHost = 5;
  *SqrtHost = 50.0f;
  *ExpInHost = 0.7f;
  *TrigHost = 0.7f;
  *FabsHost = -12;

  gpuErrCheck(KernelHandle.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr, Pow, Sqrt,
                                  Exp, Sin, Cos, Fabs, PowBaseHost, PowExpHost,
                                  SqrtHost, ExpInHost, TrigHost, FabsHost));
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
  gpuErrCheck(gpuFree(PowBaseHost));
  gpuErrCheck(gpuFree(PowExpHost));
  gpuErrCheck(gpuFree(SqrtHost));
  gpuErrCheck(gpuFree(ExpInHost));
  gpuErrCheck(gpuFree(TrigHost));
  gpuErrCheck(gpuFree(FabsHost));

  return 0;
}

// clang-format off
// CHECK: pow = 7.59375
// CHECK-NEXT: sqrt = 7.07107
// CHECK-NEXT: exp = 2.01375
// CHECK-NEXT: sin = 0.644218
// CHECK-NEXT: cos = 0.764842
// CHECK-NEXT: fabs = 12
// CHECK-FIRST: [proteus][Dispatcher{{CUDA|HIP}}] MemoryCache rank 0 hits 0 accesses 1
// CHECK: [proteus][Dispatcher{{CUDA|HIP}}] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: [proteus][Dispatcher{{CUDA|HIP}}] StorageCache rank 0 hits 0 accesses 1
// CHECK-SECOND: [proteus][Dispatcher{{CUDA|HIP}}] StorageCache rank 0 hits 1 accesses 1
