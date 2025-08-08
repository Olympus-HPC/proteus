// clang-format off
// RUN: rm -rf .proteus
// RUN: ./if.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-FIRST
// Second run uses the object cache.
// RUN: ./if.%ext | %FILECHECK %s --check-prefixes=CHECK,CHECK-SECOND
// RUN: rm -rf .proteus
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
  auto KernelHandleLT = J.addKernel<double, double, double *>("if_lt");
  auto &LT = KernelHandleLT.F;
  {
    auto &Arg0 = LT.getArg(0);
    auto &Arg1 = LT.getArg(1);
    auto &Ret = LT.getArg(2);

    LT.beginFunction();
    {
      Ret[0] = 0;
      LT.beginIf(Arg0 < Arg1);
      { Ret[0] = 1; }
      LT.endIf();
      LT.ret();
    }
    LT.endFunction();
  }

  auto KernelHandleLE = J.addKernel<double, double, double *>("if_le");
  auto &LE = KernelHandleLE.F;
  {
    auto &Arg0 = LE.getArg(0);
    auto &Arg1 = LE.getArg(1);
    auto &Ret = LE.getArg(2);

    LE.beginFunction();
    {
      Ret[0] = 0;
      LE.beginIf(Arg0 <= Arg1);
      { Ret[0] = 1; }
      LE.endIf();
      LE.ret();
    }
    LE.endFunction();
  }

  auto KernelHandleGT = J.addKernel<double, double, double *>("if_gt");
  auto &GT = KernelHandleGT.F;
  {
    auto &Arg0 = GT.getArg(0);
    auto &Arg1 = GT.getArg(1);
    auto &Ret = GT.getArg(2);

    GT.beginFunction();
    {
      Ret[0] = 0;
      GT.beginIf(Arg0 > Arg1);
      { Ret[0] = 1; }
      GT.endIf();
      GT.ret();
    }
    GT.endFunction();
  }

  auto KernelHandleGE = J.addKernel<double, double, double *>("if_ge");
  auto &GE = KernelHandleGE.F;
  {
    auto &Arg0 = GE.getArg(0);
    auto &Arg1 = GE.getArg(1);
    auto &Ret = GE.getArg(2);

    GE.beginFunction();
    {
      Ret[0] = 0;
      GE.beginIf(Arg0 >= Arg1);
      { Ret[0] = 1; }
      GE.endIf();
      GE.ret();
    }
    GE.endFunction();
  }

  auto KernelHandleEQ = J.addKernel<double, double, double *>("if_eq");
  auto &EQ = KernelHandleEQ.F;
  {
    auto &Arg0 = EQ.getArg(0);
    auto &Arg1 = EQ.getArg(1);
    auto &Ret = EQ.getArg(2);

    EQ.beginFunction();
    {
      Ret[0] = 0;
      EQ.beginIf(Arg0 == Arg1);
      { Ret[0] = 1; }
      EQ.endIf();
      EQ.ret();
    }
    EQ.endFunction();
  }

  auto KernelHandleNE = J.addKernel<double, double, double *>("if_ne");
  auto &NE = KernelHandleNE.F;
  {
    auto &Arg0 = NE.getArg(0);
    auto &Arg1 = NE.getArg(1);
    auto &Ret = NE.getArg(2);

    NE.beginFunction();
    {
      Ret[0] = 0;
      NE.beginIf(Arg0 != Arg1);
      { Ret[0] = 1; }
      NE.endIf();
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
// CHECK: JitCache hits 8 total 14
// CHECK-DAG: HashValue {{[0-9]+}} NumExecs 2 NumHits 1
// CHECK-DAG: HashValue {{[0-9]+}} NumExecs 2 NumHits 1
// CHECK-DAG: HashValue {{[0-9]+}} NumExecs 3 NumHits 2
// CHECK-DAG: HashValue {{[0-9]+}} NumExecs 2 NumHits 1
// CHECK-DAG: HashValue {{[0-9]+}} NumExecs 2 NumHits 1
// CHECK-DAG: HashValue {{[0-9]+}} NumExecs 3 NumHits 2
// CHECK-FIRST: JitStorageCache hits 0 total 1
// CHECK-SECOND: JitStorageCache hits 1 total 1
