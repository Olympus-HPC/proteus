// clang-format off
// RUN: rm -rf "%t.$$.proteus"
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/if.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: PROTEUS_CACHE_DIR="%t.$$.proteus" %build/if.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
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
  auto KernelHandleLT = J.addKernelTT<void(double, double, double *)>("if_lt");
  auto &LT = KernelHandleLT.F;
  {
    auto &Arg0 = LT.getArgTT<0>();
    auto &Arg1 = LT.getArgTT<1>();
    auto &Ret = LT.getArgTT<2>();

    LT.beginFunction();
    {
      Ret[0] = 0;
      LT.beginIfTT(Arg0 < Arg1);
      { Ret[0] = 1; }
      LT.endIfTT();
      LT.ret();
    }
    LT.endFunction();
  }

  auto KernelHandleLE = J.addKernelTT<void(double, double, double *)>("if_le");
  auto &LE = KernelHandleLE.F;
  {
    auto &Arg0 = LE.getArgTT<0>();
    auto &Arg1 = LE.getArgTT<1>();
    auto &Ret = LE.getArgTT<2>();

    LE.beginFunction();
    {
      Ret[0] = 0;
      LE.beginIfTT(Arg0 <= Arg1);
      { Ret[0] = 1; }
      LE.endIfTT();
      LE.ret();
    }
    LE.endFunction();
  }

  auto KernelHandleGT = J.addKernelTT<void(double, double, double *)>("if_gt");
  auto &GT = KernelHandleGT.F;
  {
    auto &Arg0 = GT.getArgTT<0>();
    auto &Arg1 = GT.getArgTT<1>();
    auto &Ret = GT.getArgTT<2>();

    GT.beginFunction();
    {
      Ret[0] = 0;
      GT.beginIfTT(Arg0 > Arg1);
      { Ret[0] = 1; }
      GT.endIfTT();
      GT.ret();
    }
    GT.endFunction();
  }

  auto KernelHandleGE = J.addKernelTT<void(double, double, double *)>("if_ge");
  auto &GE = KernelHandleGE.F;
  {
    auto &Arg0 = GE.getArgTT<0>();
    auto &Arg1 = GE.getArgTT<1>();
    auto &Ret = GE.getArgTT<2>();

    GE.beginFunction();
    {
      Ret[0] = 0;
      GE.beginIfTT(Arg0 >= Arg1);
      { Ret[0] = 1; }
      GE.endIfTT();
      GE.ret();
    }
    GE.endFunction();
  }

  auto KernelHandleEQ = J.addKernelTT<void(double, double, double *)>("if_eq");
  auto &EQ = KernelHandleEQ.F;
  {
    auto &Arg0 = EQ.getArgTT<0>();
    auto &Arg1 = EQ.getArgTT<1>();
    auto &Ret = EQ.getArgTT<2>();

    EQ.beginFunction();
    {
      Ret[0] = 0;
      EQ.beginIfTT(Arg0 == Arg1);
      { Ret[0] = 1; }
      EQ.endIfTT();
      EQ.ret();
    }
    EQ.endFunction();
  }

  auto KernelHandleNE = J.addKernelTT<void(double, double, double *)>("if_ne");
  auto &NE = KernelHandleNE.F;
  {
    auto &Arg0 = NE.getArgTT<0>();
    auto &Arg1 = NE.getArgTT<1>();
    auto &Ret = NE.getArgTT<2>();

    NE.beginFunction();
    {
      Ret[0] = 0;
      NE.beginIfTT(Arg0 != Arg1);
      { Ret[0] = 1; }
      NE.endIfTT();
      NE.ret();
    }
    NE.endFunction();
  }

  J.compile();

  // LT tests
  // Evaluates to true.
  double *Ret;
  gpuErrCheck(gpuMallocManaged(&Ret, sizeof(double)));
  gpuErrCheck(
      KernelHandleLT.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr, 1.0, 2.0, Ret));
  gpuErrCheck(gpuDeviceSynchronize());
  std::cout << "R LT " << *Ret << "\n";
  // Evaluates to false.
  gpuErrCheck(
      KernelHandleLT.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr, 2.0, 1.0, Ret));
  gpuErrCheck(gpuDeviceSynchronize());
  std::cout << "R LT " << *Ret << "\n";

  // LE tests
  // Evaluates to true.
  gpuErrCheck(
      KernelHandleLE.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr, 1.0, 2.0, Ret));
  gpuErrCheck(gpuDeviceSynchronize());
  std::cout << "R LE " << *Ret << "\n";
  // Evaluates to true (equal).
  gpuErrCheck(
      KernelHandleLE.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr, 2.0, 2.0, Ret));
  gpuErrCheck(gpuDeviceSynchronize());
  std::cout << "R LE " << *Ret << "\n";
  // Evaluates to false.
  gpuErrCheck(
      KernelHandleLE.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr, 3.0, 2.0, Ret));
  gpuErrCheck(gpuDeviceSynchronize());
  std::cout << "R LE " << *Ret << "\n";

  // GT tests
  // Evaluates to true.
  gpuErrCheck(
      KernelHandleGT.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr, 3.0, 2.0, Ret));
  gpuErrCheck(gpuDeviceSynchronize());
  std::cout << "R GT " << *Ret << "\n";
  // Evaluates to false.
  gpuErrCheck(
      KernelHandleGT.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr, 2.0, 3.0, Ret));
  gpuErrCheck(gpuDeviceSynchronize());
  std::cout << "R GT " << *Ret << "\n";

  // GE tests
  // Evaluates to true.
  gpuErrCheck(
      KernelHandleGE.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr, 3.0, 2.0, Ret));
  gpuErrCheck(gpuDeviceSynchronize());
  std::cout << "R GE " << *Ret << "\n";
  // Evaluates to true (equal).
  gpuErrCheck(
      KernelHandleGE.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr, 2.0, 2.0, Ret));
  gpuErrCheck(gpuDeviceSynchronize());
  std::cout << "R GE " << *Ret << "\n";
  // Evaluates to false.
  gpuErrCheck(
      KernelHandleGE.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr, 1.0, 2.0, Ret));
  gpuErrCheck(gpuDeviceSynchronize());
  std::cout << "R GE " << *Ret << "\n";

  // EQ tests
  // Evaluates to true.
  gpuErrCheck(
      KernelHandleEQ.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr, 2.0, 2.0, Ret));
  gpuErrCheck(gpuDeviceSynchronize());
  std::cout << "R EQ " << *Ret << "\n";
  // Evaluates to false.
  gpuErrCheck(
      KernelHandleEQ.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr, 2.0, 3.0, Ret));
  gpuErrCheck(gpuDeviceSynchronize());
  std::cout << "R EQ " << *Ret << "\n";

  // NE tests
  // Evaluates to true.
  gpuErrCheck(
      KernelHandleNE.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr, 2.0, 3.0, Ret));
  gpuErrCheck(gpuDeviceSynchronize());
  std::cout << "R NE " << *Ret << "\n";
  // Evaluates to false.
  gpuErrCheck(
      KernelHandleNE.launch({1, 1, 1}, {1, 1, 1}, 0, nullptr, 2.0, 2.0, Ret));
  gpuErrCheck(gpuDeviceSynchronize());
  std::cout << "R NE " << *Ret << "\n";

  gpuErrCheck(gpuFree(Ret));

  proteus::finalize();
  return 0;
}

// clang-format off
// CHECK: R LT 1
// CHECK-NEXT: R LT 0
// CHECK-NEXT: R LE 1
// CHECK-NEXT: R LE 1
// CHECK-NEXT: R LE 0
// CHECK-NEXT: R GT 1
// CHECK-NEXT: R GT 0
// CHECK-NEXT: R GE 1
// CHECK-NEXT: R GE 1
// CHECK-NEXT: R GE 0
// CHECK-NEXT: R EQ 1
// CHECK-NEXT: R EQ 0
// CHECK-NEXT: R NE 1
// CHECK-NEXT: R NE 0
// NOTE: Cache hits are less because the KernelHandle stores the kernel function, avoiding the cache.
// CHECK: JitCache hits 0 total 6
// CHECK-COUNT-6: HashValue {{[0-9]+}} NumExecs 1 NumHits 0
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
