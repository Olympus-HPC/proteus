// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/operators.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/operators.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf "%t.$$.proteus"
// clang-format on

#include <iostream>

#include <proteus/JitFrontend.h>
#include <proteus/JitInterface.h>

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
  auto KernelHandle = J.addKernel<void(
      const double *, const double *, double *, double *, double *, double *,
      double *, double *, double *, double *, double *, double *, double *,
      double *, double *, double *, double *)>("operators");
  auto &F = KernelHandle.F;
  auto &Arg0 = F.getArg<0>();
  auto &Arg1 = F.getArg<1>();
  auto &Arg2 = F.getArg<2>();
  auto &Arg3 = F.getArg<3>();
  auto &Arg4 = F.getArg<4>();
  auto &Arg5 = F.getArg<5>();
  auto &Arg6 = F.getArg<6>();
  auto &Arg7 = F.getArg<7>();
  auto &Arg8 = F.getArg<8>();
  auto &Arg9 = F.getArg<9>();
  auto &Arg10 = F.getArg<10>();
  auto &Arg11 = F.getArg<11>();
  auto &Arg12 = F.getArg<12>();
  auto &Arg13 = F.getArg<13>();
  auto &Arg14 = F.getArg<14>();
  auto &Arg15 = F.getArg<15>();
  auto &Arg16 = F.getArg<16>();
  F.beginFunction();
  {
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

    auto Cmp = F.declVar<double>("cmp");
    Cmp = 5.0;
    Arg11[0] = 0.0;
    F.beginIf(Cmp <= 5.0);
    { Arg11[0] = 1.0; }
    F.endIf();

    auto ConstVal = F.defVar<const int>(Cmp, "const_val");
    Arg10[0] = Arg10[0] + ConstVal;

    auto ConstValScalar = F.defVar<const int>(1, "const_val_scalar");
    Arg10[0] = Arg10[0] + ConstValScalar;

    Arg12[0] = -Arg0[0];

    auto NotCond = !(Cmp <= 5.0);
    Arg13[0] = NotCond;

    // Test reference semantics: store dereference result in auto variable.
    // This tests that operator*() returns Var<T&> which works correctly.
    auto Ref14 = *Arg14;
    Ref14 = 42.0;

    // Test reference semantics: store subscript result in auto variable.
    // This tests that operator[] returns Var<T&> which works correctly.
    auto Ref15 = Arg15[0];
    Ref15 = 43.0;

    // Test that arithmetic on reference types works correctly.
    // The result should be Var<T> (not Var<T&>).
    auto Ref16 = *Arg16;
    Ref16 = Ref14 + Ref15;

    F.ret();
  }
  F.endFunction();

  J.compile();

  double *R0, *R1, *R2, *R3, *R4, *R5, *R6, *R7, *R8, *R9, *R10, *R11, *R12,
      *R13, *R14, *R15, *R16;
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
  gpuErrCheck(gpuMallocManaged(&R12, sizeof(double)));
  gpuErrCheck(gpuMallocManaged(&R13, sizeof(double)));
  gpuErrCheck(gpuMallocManaged(&R14, sizeof(double)));
  gpuErrCheck(gpuMallocManaged(&R15, sizeof(double)));
  gpuErrCheck(gpuMallocManaged(&R16, sizeof(double)));

  *R0 = 2.0;
  *R1 = 3.0;
  *R14 = *R15 = *R16 = 0.0;

  gpuErrCheck(KernelHandle.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr, R0, R1, R2,
                                  R3, R4, R5, R6, R7, R8, R9, R10, R11, R12,
                                  R13, R14, R15, R16));
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
  std::cout << "R12 = " << *R12 << "\n";
  std::cout << "R13 = " << *R13 << "\n";
  std::cout << "R14 = " << *R14 << "\n";
  std::cout << "R15 = " << *R15 << "\n";
  std::cout << "R16 = " << *R16 << "\n";

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
  gpuErrCheck(gpuFree(R12));
  gpuErrCheck(gpuFree(R13));
  gpuErrCheck(gpuFree(R14));
  gpuErrCheck(gpuFree(R15));
  gpuErrCheck(gpuFree(R16));

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
// CHECK-NEXT: R10 = 13
// CHECK-NEXT: R11 = 1
// CHECK-NEXT: R12 = -2
// CHECK-NEXT: R13 = 0
// CHECK-NEXT: R14 = 42
// CHECK-NEXT: R15 = 43
// CHECK-NEXT: R16 = 85
// CHECK-NEXT: proteus][Dispatcher{{CUDA|HIP}}] MemoryCache rank 0 hits 0 accesses 1
// CHECK-NEXT: [proteus][Dispatcher{{CUDA|HIP}}] MemoryCache rank 0 HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: [proteus][Dispatcher{{CUDA|HIP}}] StorageCache rank 0 hits 0 accesses 1
// CHECK-SECOND: [proteus][Dispatcher{{CUDA|HIP}}] StorageCache rank 0 hits 1 accesses 1
