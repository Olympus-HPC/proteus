// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/operators.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/operators.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
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
  auto KernelHandle = J.addKernelTT<void(
      double *, double *, double *, double *, double *, double *, double *,
      double *, double *, double *, double *, double *)>("operators");
  auto &F = KernelHandle.F;
    auto &Arg0 = F.getArgTT<0>();
    auto &Arg1 = F.getArgTT<1>();
    auto &Arg2 = F.getArgTT<2>();
    auto &Arg3 = F.getArgTT<3>();
    auto &Arg4 = F.getArgTT<4>();
    auto &Arg5 = F.getArgTT<5>();
    auto &Arg6 = F.getArgTT<6>();
    auto &Arg7 = F.getArgTT<7>();
    auto &Arg8 = F.getArgTT<8>();
    auto &Arg9 = F.getArgTT<9>();
    auto &Arg10 = F.getArgTT<10>();
    auto &Arg11 = F.getArgTT<11>();
  F.beginFunction();
  {
    Arg0[0] = 2;
    Arg1[0] = 3;

    Arg2[0] = Arg0[0] + Arg1[0];
    Arg3[0] = Arg0[0] - Arg1[0];
    Arg4[0] = Arg0[0] * Arg1[0];
    Arg5[0] = Arg0[0] / Arg1[0];

    *Arg6 = *Arg7 = *Arg8 = *Arg9 = 5;

    Arg6[0] += Arg0[0];
    Arg7[0] -= Arg0[0];
    Arg8[0] *= Arg0[0];
    Arg9[0] /= Arg0[0];

    Arg10[0] = 10.0;
    Arg10[0] -= 3.0;

    auto Cmp = F.declVarTT<double>("cmp");
    Cmp = 5.0;
    Arg11[0] = 0.0;
    F.beginIfTT(Cmp <= 5.0);
    { Arg11[0] = 1.0; }
    F.endIfTT();

    F.retTT();
  }
  F.endFunction();

  J.compile();

  double *R0, *R1, *R2, *R3, *R4, *R5, *R6, *R7, *R8, *R9, *R10, *R11;
  gpuErrCheck(gpuMallocManaged(&R0, sizeof(double)));
  gpuErrCheck(gpuMallocManaged(&R1, sizeof(double)));
  gpuErrCheck(gpuMallocManaged(&R2, sizeof(double)));
  gpuErrCheck(gpuMallocManaged(&R3, sizeof(double)));
  gpuErrCheck(gpuMallocManaged(&R4, sizeof(double)));
  gpuErrCheck(gpuMallocManaged(&R5, sizeof(double)));
  gpuErrCheck(gpuMallocManaged(&R6, sizeof(double)));
  gpuErrCheck(gpuMallocManaged(&R7, sizeof(double)));
  gpuErrCheck(gpuMallocManaged(&R8, sizeof(double)));
  gpuErrCheck(gpuMallocManaged(&R9, sizeof(double)));
  gpuErrCheck(gpuMallocManaged(&R10, sizeof(double)));
  gpuErrCheck(gpuMallocManaged(&R11, sizeof(double)));

  gpuErrCheck(KernelHandle.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr, R0, R1, R2,
                                  R3, R4, R5, R6, R7, R8, R9, R10, R11));
  gpuErrCheck(gpuDeviceSynchronize());

  std::cout << "R0 = " << *R0 << "\n";
  std::cout << "R1 = " << *R1 << "\n";
  std::cout << "R2 = " << *R2 << "\n";
  std::cout << "R3 = " << *R3 << "\n";
  std::cout << "R4 = " << *R4 << "\n";
  std::cout << "R5 = " << *R5 << "\n";
  std::cout << "R6 = " << *R6 << "\n";
  std::cout << "R7 = " << *R7 << "\n";
  std::cout << "R8 = " << *R8 << "\n";
  std::cout << "R9 = " << *R9 << "\n";
  std::cout << "R10 = " << *R10 << "\n";
  std::cout << "R11 = " << *R11 << "\n";

  gpuErrCheck(gpuFree(R0));
  gpuErrCheck(gpuFree(R1));
  gpuErrCheck(gpuFree(R2));
  gpuErrCheck(gpuFree(R3));
  gpuErrCheck(gpuFree(R4));
  gpuErrCheck(gpuFree(R5));
  gpuErrCheck(gpuFree(R6));
  gpuErrCheck(gpuFree(R7));
  gpuErrCheck(gpuFree(R8));
  gpuErrCheck(gpuFree(R9));
  gpuErrCheck(gpuFree(R10));
  gpuErrCheck(gpuFree(R11));

  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK: R0 = 2
// CHECK-NEXT: R1 = 3
// CHECK-NEXT: R2 = 5
// CHECK-NEXT: R3 = -1
// CHECK-NEXT: R4 = 6
// CHECK-NEXT: R5 = 0.666667
// CHECK-NEXT: R6 = 7
// CHECK-NEXT: R7 = 3
// CHECK-NEXT: R8 = 10
// CHECK-NEXT: R9 = 2.5
// CHECK-NEXT: R10 = 7
// CHECK-NEXT: R11 = 1
// CHECK-NEXT: JitCache hits 0 total 1
// CHECK-NEXT: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
